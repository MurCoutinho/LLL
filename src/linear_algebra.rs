///This trait is designed to implement basic linear algebra functionalities to base types.
pub trait VecLinearAlgebra<T> {
    ///The dot product between two vectors.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let a: Vec<f64> = vec![1.0, 2.0];
    /// let b: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(a.dot(&b), 11.0);
    /// ~~~
    fn dot(&self, v: &[T]) -> T;

    ///Computes the squared norm of the vector.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let v: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(v.norm_squared(), 25.0);
    /// ~~~
    fn norm_squared(&self) -> T;

    ///Computes the norm of the vector.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let v: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(v.norm(), 5.0);
    /// ~~~
    fn norm(&self) -> f64;

    ///Adds two vectors.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let a: Vec<f64> = vec![1.0, 2.0];
    /// let b: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(a.add(&b), vec![4.0, 6.0]);
    /// ~~~
    fn add(&self, v: &[T]) -> Vec<T>;

    ///Adds two vectors.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let a: Vec<f64> = vec![1.0, 2.0];
    /// let b: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(a.sub(&b), vec![-2.0, -2.0]);
    /// ~~~
    fn sub(&self, v: &[T]) -> Vec<T>;

    ///Multiplies a vector by a scalar.
    ///
    /// # Examples
    ///
    /// ~~~
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let v: Vec<f64> = vec![3.0, 4.0];
    /// assert_eq!(v.scalar_mult(5.0), vec![15.0, 20.0]);
    /// ~~~
    fn scalar_mult(&self, a: T) -> Vec<T>;

    ///Computes the projection of the vector into a space spanned by some basis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lattice_cryptanalysis::linear_algebra::VecLinearAlgebra;
    /// let basis = vec![vec![1.0, 2.0, 2.0], vec![2.0, 1.0, -2.0]];
    /// let v = vec![2.0, 9.0, -4.0];
    /// println!("{:?}", v.projection(&basis));
    /// ```
    fn projection(&self, basis: &[Vec<T>]) -> Vec<f64>;
}

pub trait MatLinearAlgebra<T> {
    /// Compute the transpose of a matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use lattice_cryptanalysis::linear_algebra::MatLinearAlgebra;
    /// let m = vec![vec![1.0, 2.0, 3.0],vec![4.0, 5.0, 6.0],];
    /// let mt = vec![vec![1.0,4.0],vec![2.0,5.0],vec![3.0,6.0],];
    /// assert_eq!(mt, m.transpose());
    /// ```
    fn transpose(&self) -> Vec<Vec<T>>;

    fn mat_mult(&self, m: &Vec<Vec<T>>) -> Vec<Vec<T>>;
}

//Implementation of basic linear algebra methods for f64.
impl VecLinearAlgebra<f64> for Vec<f64> {
    fn dot(&self, v: &[f64]) -> f64 {
        self.iter().zip(v.iter()).map(|(x, y)| x * y).sum::<f64>()
    }

    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum::<f64>()
    }

    fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    fn add(&self, v: &[f64]) -> Vec<f64> {
        self.iter().zip(v.iter()).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, v: &[f64]) -> Vec<f64> {
        self.iter().zip(v.iter()).map(|(x, y)| x - y).collect()
    }

    fn scalar_mult(&self, a: f64) -> Vec<f64> {
        self.iter().map(|x| x * a).collect()
    }

    fn projection(&self, basis: &[Vec<f64>]) -> Vec<f64> {
        let mut new_vec = vec![0.0; self.len()];
        for v in basis.iter() {
            new_vec = new_vec.add(&v.scalar_mult(self.dot(v) / v.norm_squared()));
        }
        new_vec
    }
}

impl VecLinearAlgebra<i128> for Vec<i128> {
    fn dot(&self, v: &[i128]) -> i128 {
        self.iter().zip(v.iter()).map(|(x, y)| x * y).sum::<i128>()
    }

    fn norm_squared(&self) -> i128 {
        self.iter().map(|x| x * x).sum::<i128>()
    }

    fn norm(&self) -> f64 {
        (self.norm_squared() as f64).sqrt()
    }

    fn add(&self, v: &[i128]) -> Vec<i128> {
        self.iter().zip(v.iter()).map(|(x, y)| x + y).collect()
    }

    fn sub(&self, v: &[i128]) -> Vec<i128> {
        self.iter().zip(v.iter()).map(|(x, y)| x - y).collect()
    }

    fn scalar_mult(&self, a: i128) -> Vec<i128> {
        self.iter().map(|x| x * a).collect()
    }

    fn projection(&self, basis: &[Vec<i128>]) -> Vec<f64> {
        let mut new_vec = vec![0.0; self.len()];
        for v in basis.iter() {
            let vf = v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
            new_vec = new_vec.add(&vf.scalar_mult((self.dot(v) / v.norm_squared()) as f64));
        }
        new_vec
    }
}

impl MatLinearAlgebra<f64> for Vec<Vec<f64>> {
    fn transpose(&self) -> Vec<Vec<f64>> {
        let mut t = vec![Vec::with_capacity(self.len()); self[0].len()];
        for r in self {
            for i in 0..r.len() {
                t[i].push(r[i]);
            }
        }
        t
    }

    fn mat_mult(&self, m: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mt = m.transpose();
        let mut r = vec![Vec::with_capacity(self.len()); self[0].len()];
        for (i, x) in self.iter().enumerate() {
            for y in mt.iter() {
                r[i].push(x.dot(y));
            }
        }
        r
    }
}

/// The Gram Schmidt algorithm computes an orthogonal basis given an arbitrary basis.
///
/// # Examples
/// ```
/// # use lattice_cryptanalysis::linear_algebra::{gram_schmidt, VecLinearAlgebra};
/// let basis = vec![vec![1.0, 2.0],vec![3.0, 7.0]];
/// let orth_basis = gram_schmidt(&basis);
/// assert_eq!(orth_basis[0].dot(&orth_basis[1]).round(), 0.0);
/// ```
pub fn gram_schmidt(basis: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut new_basis = vec![basis[0].clone()];
    for v in basis.iter().skip(1) {
        new_basis.push(v.sub(&v.projection(&new_basis)));
    }

    new_basis
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_projection() {
        let basis = vec![vec![1.0, 2.0, 2.0], vec![2.0, 1.0, -2.0]];
        let v = vec![2.0, 9.0, -4.0];

        assert_eq!(
            v.projection(&basis)
                .iter()
                .map(|x| x.round())
                .collect::<Vec<f64>>(),
            vec![6.0, 5.0, -2.0]
        );
    }

    #[test]
    fn test_gram_schmidt() {
        let basis = vec![
            vec![1.0, 2.0, 3.0, 11.0],
            vec![3.0, 7.0, 6.0, 12.0],
            vec![5.0, 2.0, 3.0, -1.0],
            vec![7.0, 2.0, -1.0, -5.0],
        ];

        let orth_basis = gram_schmidt(&basis);
        for i in 0..(orth_basis.len() - 1) {
            for j in (i + 1)..orth_basis.len() {
                println!("{} {}", i, j);
                assert_eq!(orth_basis[i].dot(&orth_basis[j]).round(), 0.0);
            }
        }
    }

    #[test]
    fn test_transpose() {
        let m = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![3.0, 5.0, 6.0],
        ];

        let mt = vec![
            vec![1.0, -1.0, 3.0],
            vec![1.0, 0.0, 5.0],
            vec![1.0, 2.0, 6.0],
        ];

        assert_eq!(mt, m.transpose());
        assert_eq!(m, m.transpose().transpose());
    }

    #[test]
    fn test_mat_mult() {
        let h = vec![
            vec![-4.0, -1.0, 1.0],
            vec![5.0, 1.0, -1.0],
            vec![-5.0, 0.0, 1.0],
        ];

        let b = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![3.0, 5.0, 6.0],
        ];

        let r = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![-2.0, 0.0, 1.0],
        ];

        assert_eq!(r, h.mat_mult(&b));
    }
}
