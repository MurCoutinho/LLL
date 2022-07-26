use criterion::{criterion_group, criterion_main, Criterion};
use lattice_cryptanalysis::lattice::{lll, Lattice};
use rand::Rng;
use std::{fmt::Write, time::Duration};

pub fn bench_lll(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let sizes = [10, 20, 40, 80];
    for n in sizes {
        let mut basis: Vec<Vec<i128>> = Vec::new();
        (0..n).for_each(|_| {
            let mut v: Vec<i128> = Vec::new();
            (0..n).for_each(|_| v.push(rng.gen_range(-500..500)));
            basis.push(v);
        });
        let l = Lattice::from_integral_basis(basis);

        let mut text = String::new();
        write!(&mut text, "LLL bench for n = {}!", n).unwrap();

        let mut group = c.benchmark_group("LLL bench");
        group.measurement_time(Duration::new(30, 0));
        group.bench_function(&text, |b| b.iter(|| lll(&l)));
        group.finish();
    }
}

criterion_group!(benches, bench_lll);
criterion_main!(benches);
