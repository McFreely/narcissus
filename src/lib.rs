#![crate_name = "narcissus"]
#![feature(test)]

extern crate test;

use std::cmp;

pub fn euclidian_distance(vec_one: Vec<f32>, vec_two: Vec<f32>) -> f32 {
    let mut result: f32 = 0.0;

    for (i, j) in vec_one.iter().zip(vec_two) {
        result += (i - j).abs().powf(2.0);
    }

    result.sqrt()
}

pub fn cosine_similarity(vec_one: Vec<f32>, vec_two: Vec<f32>) -> f32 {
    let dot_product = dot_product_optim(&vec_one, &vec_two);
    let magnitude = magnitude(&vec_one) * magnitude(&vec_two);

    dot_product / magnitude
}

pub fn jaccard_coeficient(vec_one: Vec<f32>, vec_two: Vec<f32>) -> f32 {
    let dot_product = dot_product_optim(&vec_one, &vec_two);
    let magnitude_vec_one = magnitude(&vec_one);
    let magnitude_vec_two = magnitude(&vec_two);

    dot_product / ((magnitude_vec_one.abs().powf(2.0) + magnitude_vec_two.abs().powf(2.0)) - (dot_product))
}

// pub fn pearson_correlation<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
// }

// pub fn kullback_liebler_divergence<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
// }

pub fn dot_product_naive(xs: &Vec<f32>, ys: &Vec<f32>) -> f32 {
    let mut result: f32 = 0.0;

    for (x, y) in xs.iter().zip(ys) {
        result += x * y
    }

    result
}

pub fn dot_product_optim(xs: &Vec<f32>, ys: &Vec<f32>) -> f32 {
    let mut result: f32 = 0.0;

    let len = cmp::min(xs.len(), ys.len());
    let xs = &xs[..len];
    let ys = &ys[..len];

    for i in 0..len {
        result += &xs[i] * &ys[i]
    }

    result
}

fn magnitude(vec: &Vec<f32>) -> f32 {
    // The magnitude of a vector is the sqrt of its own dotproduct
    dot_product_optim(vec, vec).sqrt()
}

#[cfg(test)]
mod tests {

    use super::test;
    use super::euclidian_distance;
    use super::cosine_similarity as cosine;
    use super::jaccard_coeficient as jaccard;
    use super::dot_product_naive;
    use super::dot_product_optim;

    use test::Bencher;
    use std::f32;

    #[test]
    fn euclidian_distance_works() {
        let vec_one = vec![1.0, 2.0, 3.0];
        let vec_two = vec![2.0, 1.0, 4.0];

        let distance = euclidian_distance(vec_one, vec_two);

        assert!((distance - 1.7320508).abs() <= f32::EPSILON); 
    }

    #[test]
    fn cosine_similarity() {
        let vec_one = vec![2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0];
        let vec_two = vec![2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];

        let sim = cosine(vec_one, vec_two);

        assert_eq!(sim, 0.8215838);
    }

    #[test]
    fn jaccard_coeficient() {
        let vec_one = vec![2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0];
        let vec_two = vec![2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        
        let sim = jaccard(vec_one, vec_two);

        assert_eq!(sim, 0.6923077);
    }

    #[bench]
    fn bench_naive_dot_product(b: &mut Bencher) {
        let vec_one = vec![1.2; 10292];
        let vec_two = vec![0.3; 8930];

        b.iter(|| dot_product_naive(&vec_one, &vec_two));
    }

    #[bench]
    fn bench_optim_dot_product(b: &mut Bencher) {
        let vec_one = vec![1.2; 10292];
        let vec_two = vec![0.3; 8930];

        b.iter(|| dot_product_optim(&vec_one, &vec_two));
    }
}
