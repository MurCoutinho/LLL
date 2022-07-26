use lattice_cryptanalysis::lattice::{lll, Lattice};
use std::vec;

fn main() {
    //Example
    let m = vec![
        vec![19.0, 2.0, 32.0, 46.0, 3.0, 33.0],
        vec![15.0, 42.0, 11.0, 0.0, 3.0, 24.0],
        vec![43.0, 15.0, 0.0, 24.0, 4.0, 16.0],
        vec![20.0, 44.0, 44.0, 0.0, 18.0, 15.0],
        vec![0.0, 48.0, 35.0, 16.0, 31.0, 31.0],
        vec![48.0, 33.0, 32.0, 9.0, 1.0, 29.0],
    ];

    let l = Lattice::new(m);
    println!("{:?}", lll(&l));
}
