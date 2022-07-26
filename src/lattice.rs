use crate::linear_algebra::VecLinearAlgebra;
use thiserror::Error;

///Defines a lattice
#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub basis: Vec<Vec<f64>>,
}

impl Lattice {
    pub fn new(basis: Vec<Vec<f64>>) -> Self {
        Self { basis }
    }

    pub fn from_integral_basis(basis: Vec<Vec<i128>>) -> Self {
        Self {
            basis: basis
                .iter()
                .map(|v| v.iter().map(|x| *x as f64).collect::<Vec<f64>>())
                .collect::<Vec<Vec<f64>>>(),
        }
    }

    pub fn is_integral(&self) -> bool {
        self.basis.iter().flatten().fold(0.0, |_acc, x| x.fract()) == 0.0
    }

    pub fn get_basis_as_integer(&self) -> Result<Vec<Vec<i128>>, LatticeError> {
        if !self.is_integral() {
            return Err(LatticeError::NotIntegral);
        }

        Ok(self
            .basis
            .iter()
            .map(|v| v.iter().map(|x| *x as i128).collect::<Vec<i128>>())
            .collect::<Vec<Vec<i128>>>())
    }

    pub fn get_min_norm_from_basis(&self) -> f64 {
        self.basis
            .iter()
            .map(|x| x.norm())
            .collect::<Vec<f64>>()
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
    }
}

///Possible errors when dealing with lattice functions.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum LatticeError {
    ///The dimension of this lattices is not appropriate for a particular method or function.
    #[error("The dimension of this lattices is not appropriate for this method or function.")]
    InvalidDimension,
    #[error("Sorry, the informed vectors do not form a basis.")]
    NotBasis,
    #[error("The lattice is not integral.")]
    NotIntegral,
}

///This function implements the Gaussian lattice reduction and solves the SVP problem for dimension 2 lattices.
///The algorithm for finding an optimal basis in a lattice of dimension 2 is essentially due to Gauss.
///
/// # Examples
///
/// ```
/// # use lattice_cryptanalysis::lattice::Lattice;
/// # use lattice_cryptanalysis::lattice::gaussian_lattice_reduction;
/// let lattice = Lattice::new(vec![
///   vec![66586820.0, 65354729.0],
///   vec![6513996.0, 6393464.0],
/// ]);
/// let answer = Lattice::new(vec![vec![2280.0, -1001.0], vec![-1324.0, -2376.0]]);
/// assert_eq!(gaussian_lattice_reduction(&lattice).unwrap(), answer);
/// ```
pub fn gaussian_lattice_reduction(lat: &Lattice) -> Result<Lattice, LatticeError> {
    let mut l = (*lat).clone();
    let (mut i, mut j): (usize, usize);
    let mut m: f64;

    if l.basis.len() != 2 {
        return Err(LatticeError::InvalidDimension);
    }

    loop {
        (i, j) = if l.basis[1].norm() < l.basis[0].norm() {
            (1, 0)
        } else {
            (0, 1)
        };

        m = l.basis[0].dot(&l.basis[1]);
        m /= l.basis[i].norm_squared();
        m = m.round();
        if m == 0.0 {
            break;
        }
        l.basis[j] = l.basis[j].sub(&l.basis[i].scalar_mult(m));
    }

    Ok(l)
}

fn lll_red(k: usize, l: usize, mu_matrix: &mut [Vec<f64>], basis: &mut [Vec<f64>]) {
    if mu_matrix[k][l].abs() > 0.5 {
        let q = mu_matrix[k][l].round();
        basis[k] = basis[k].sub(&basis[l].scalar_mult(q));
        mu_matrix[k][l] -= q;
        for i in 0..l {
            mu_matrix[k][i] -= q * mu_matrix[l][i];
        }
    }
}

fn lll_swap(
    k: usize,
    k_max: usize,
    mu_matrix: &mut [Vec<f64>],
    basis: &mut [Vec<f64>],
    basis_star: &mut [Vec<f64>],
    inner_vector: &mut [f64],
) {
    let aux = basis[k].clone();
    basis[k] = basis[k - 1].clone();
    basis[k - 1] = aux;

    if k > 1 {
        for j in 0..(k - 1) {
            let aux = mu_matrix[k][j];
            mu_matrix[k][j] = mu_matrix[k - 1][j];
            mu_matrix[k - 1][j] = aux;
        }
    }

    let m = mu_matrix[k][k - 1];
    let new_value = inner_vector[k] + m * m * inner_vector[k - 1];
    mu_matrix[k][k - 1] = m * inner_vector[k - 1] / new_value;
    let b = basis_star[k - 1].clone();
    basis_star[k - 1] = basis_star[k].add(&b.scalar_mult(m));
    basis_star[k] = b
        .scalar_mult(inner_vector[k] / new_value)
        .sub(&basis_star[k].scalar_mult(mu_matrix[k][k - 1]));

    inner_vector[k] = inner_vector[k - 1] * inner_vector[k] / new_value;
    inner_vector[k - 1] = new_value;

    for i in (k + 1)..(k_max + 1) {
        let t = mu_matrix[i][k];
        mu_matrix[i][k] = mu_matrix[i][k - 1] - m * t;
        mu_matrix[i][k - 1] = t + mu_matrix[k][k - 1] * mu_matrix[i][k];
    }
}

fn lovasz_condition(k: usize, lambda: f64, norm_vector: &[f64], mu_matrix: &[Vec<f64>]) -> bool {
    norm_vector[k] < (lambda - mu_matrix[k][k - 1] * mu_matrix[k][k - 1]) * norm_vector[k - 1]
}

/// The Lenstra, Lenstra and Lovasz (LLL) algorithm. It can be used to reduce a Lattice basis and to try to solve the SVP problem.
/// Implementation based on Alg 2.6.3 from Henri Cohen - A Course in Computational Algebraic Number Theory.
///
/// # Example
///
/// ```
/// # use lattice_cryptanalysis::lattice::{lll,Lattice};
/// let lat = Lattice::new(vec![vec![1.0, 1.0, 1.0],vec![-1.0, 0.0, 2.0],vec![3.0, 5.0, 6.0],]);
/// let ans = Lattice::new(vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![-2.0, 0.0, 1.0]]);
/// assert_eq!(ans, lll(&lat).unwrap());
/// ```
pub fn lll(lat: &Lattice) -> Result<Lattice, LatticeError> {
    let mut k: usize = 1;
    let mut k_max: usize = 0;
    let n = lat.basis.len();
    let mut basis = lat.basis.clone();
    let mut basis_star = lat.basis.clone();
    let mut mu_matrix = vec![vec![0.0; n]; n];
    let mut inner_vector: Vec<f64> = vec![0.0; n];

    inner_vector[0] = basis[0].norm_squared();

    while k < n {
        if k > k_max {
            k_max = k;
            basis_star[k] = basis[k].clone();
            for j in 0..k {
                mu_matrix[k][j] = basis[k].dot(&basis_star[j]) / inner_vector[j];
                basis_star[k] = basis_star[k].sub(&basis_star[j].scalar_mult(mu_matrix[k][j]));
            }
            inner_vector[k] = basis_star[k].norm_squared();
            if inner_vector[k] == 0.0 {
                return Err(LatticeError::NotBasis);
            }
        }

        lll_red(k, k - 1, &mut mu_matrix, &mut basis);

        if lovasz_condition(k, 0.75, &inner_vector, &mu_matrix) {
            lll_swap(
                k,
                k_max,
                &mut mu_matrix,
                &mut basis,
                &mut basis_star,
                &mut inner_vector,
            );
            k = std::cmp::max(1, k - 1);
        } else {
            (0..(k - 1)).rev().for_each(|l| {
                lll_red(k, l, &mut mu_matrix, &mut basis);
            });
            k += 1;
        }
    }

    Ok(Lattice::new(basis))
}

fn int_lll_red(
    k: usize,
    l: usize,
    mu_matrix: &mut [Vec<i128>],
    basis: &mut [Vec<i128>],
    d: &mut [i128],
) {
    if 2 * mu_matrix[k][l].abs() > d[l + 1] {
        let q = ((mu_matrix[k][l] as f64) / (d[l + 1] as f64)).round() as i128;
        basis[k] = basis[k].sub(&basis[l].scalar_mult(q));
        mu_matrix[k][l] -= q * d[l + 1];
        for i in 0..l {
            mu_matrix[k][i] -= q * mu_matrix[l][i];
        }
    }
}

fn int_lll_swap(
    k: usize,
    k_max: usize,
    mu_matrix: &mut [Vec<i128>],
    basis: &mut [Vec<i128>],
    d: &mut [i128],
) {
    let aux = basis[k].clone();
    basis[k] = basis[k - 1].clone();
    basis[k - 1] = aux;

    if k > 1 {
        for j in 0..(k - 1) {
            let aux = mu_matrix[k][j];
            mu_matrix[k][j] = mu_matrix[k - 1][j];
            mu_matrix[k - 1][j] = aux;
        }
    }

    let m = mu_matrix[k][k - 1];
    let new_value = (d[k + 1] * d[k - 1] + m * m) / d[k];

    for v in mu_matrix.iter_mut().take(k_max + 1).skip(k + 1) {
        let t = v[k];
        v[k] = (v[k - 1] * d[k + 1] - m * t) / d[k];
        v[k - 1] = (new_value * t + m * v[k]) / d[k + 1];
    }

    d[k] = new_value;
}

/// The Lenstra, Lenstra and Lovasz (LLL) algorithm when we have a basis only with integers.
/// It can be used to reduce a Lattice basis and to try to solve the SVP problem.
/// Implementation based on Alg 2.6.7 from Henri Cohen - A Course in Computational Algebraic Number Theory.
/// Original algorithm belongs to B.M.M. de Weger - Algorithms for diophantine equations (1988)
pub fn int_lll(lat: &Lattice) -> Result<Lattice, LatticeError> {
    let mut k: usize = 1;
    let mut k_max: usize = 0;
    let n = lat.basis.len();
    let mut basis = lat.get_basis_as_integer()?;
    let mut mu_matrix = vec![vec![0; n]; n];
    let mut d: Vec<i128> = vec![0; n + 1];

    d[0] = 1;
    d[1] = basis[0].norm_squared();

    while k < n {
        if k > k_max {
            k_max = k;
            for j in 0..(k + 1) {
                let mut u = basis[k].dot(&basis[j]);
                for i in 0..j {
                    u = (d[i + 1] * u - mu_matrix[k][i] * mu_matrix[j][i]) / d[i];
                }
                if j < k {
                    mu_matrix[k][j] = u;
                } else {
                    d[k + 1] = u;
                }
            }
            if d[k + 1] == 0 {
                return Err(LatticeError::NotBasis);
            }
        }

        int_lll_red(k, k - 1, &mut mu_matrix, &mut basis, &mut d);

        if d[k + 1] * d[k - 1] < (3 * d[k] * d[k]) / 4 - mu_matrix[k][k - 1] * mu_matrix[k][k - 1] {
            int_lll_swap(k, k_max, &mut mu_matrix, &mut basis, &mut d);
            k = std::cmp::max(1, k - 1);
        } else {
            (0..(k - 1)).rev().for_each(|l| {
                int_lll_red(k, l, &mut mu_matrix, &mut basis, &mut d);
            });
            k += 1;
        }
    }

    Ok(Lattice::from_integral_basis(basis))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn gaussian_lattice_reduction_silverman_example() {
        let lattice =
            Lattice::from_integral_basis(vec![vec![66586820, 65354729], vec![6513996, 6393464]]);
        let answer = Lattice::from_integral_basis(vec![vec![2280, -1001], vec![-1324, -2376]]);
        assert_eq!(gaussian_lattice_reduction(&lattice).unwrap(), answer);
    }

    #[test]
    fn gaussian_lattice_reduction_lattice_dimension() {
        let lattice = Lattice::from_integral_basis(vec![vec![2, 5], vec![3, 4], vec![1, 2]]);

        match gaussian_lattice_reduction(&lattice) {
            Ok(_) => panic!("This should not happen! Method accepted invalid dimension!"),
            Err(e) => assert_eq!(e, LatticeError::InvalidDimension),
        }
    }

    #[test]
    fn lll_test_silverman_example() {
        let answer = Lattice::from_integral_basis(vec![
            vec![7, -12, -8, 4, 19, 9],
            vec![-20, 4, -9, 16, 13, 16],
            vec![5, 2, 33, 0, 15, -9],
            vec![-6, -7, -20, -21, 8, -12],
            vec![-10, -24, 21, -15, -6, -11],
            vec![7, 4, -9, -11, 1, 31],
        ]);

        let silverman_lat = Lattice::from_integral_basis(vec![
            vec![19, 2, 32, 46, 3, 33],
            vec![15, 42, 11, 0, 3, 24],
            vec![43, 15, 0, 24, 4, 16],
            vec![20, 44, 44, 0, 18, 15],
            vec![0, 48, 35, 16, 31, 31],
            vec![48, 33, 32, 9, 1, 29],
        ]);

        assert_eq!(answer, lll(&silverman_lat).unwrap());
        assert_eq!(answer, int_lll(&silverman_lat).unwrap());
    }
}
