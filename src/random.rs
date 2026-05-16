use rand_distr::{Distribution, Normal};

pub struct RandN {
    mean: f64,
    stddev: f64,
}

impl RandN {
    pub fn new(mean: f64, stddev: f64) -> Self {
        RandN { mean, stddev }
    }

    pub fn sample(&self, ptr: &mut [f32]) {
        let normal = Normal::new(self.mean, self.stddev).unwrap();
        let mut rng = rand::thread_rng();
        for val in ptr.iter_mut() {
            *val = normal.sample(&mut rng) as f32;
        }
    }
}