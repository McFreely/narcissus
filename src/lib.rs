#![crate_name = "narcissus"]
#![feature(test)]

extern crate test;
use std::cmp;
use std::collections::HashSet;

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

pub fn matching_words_frequency(s1: &String, s2: &String) -> usize {
    // Takes two sentences as input
    // output number of matching words

    let sen_one = s1.split_whitespace().collect::<HashSet<&str>>();
    let sen_two = s2.split_whitespace().collect::<HashSet<&str>>();

    sen_one.intersection(&sen_two).count()
}

pub fn matching_words_count(s1: &String, s2: &String) -> usize {
    let mut result: usize = 0;
    let words_one = s1.split_whitespace().collect::<Vec<&str>>();
    let words_two = s2.split_whitespace().collect::<Vec<&str>>();

    for (i, j) in words_one.iter().zip(words_two) {
        if i.to_string() == j.to_string() { result += 1 }
    }

    result
}

pub enum Matching {
    Frequency,
    Count,
}

pub fn matching_sim(s1: &String, s2: &String, matching: Matching) -> usize {
    let mwc = match matching {
        Matching::Frequency => matching_words_frequency(&s1, &s2),
        Matching::Count => matching_words_count(&s1, &s2),
    };

    (2 * mwc) / (s1.split_whitespace().count() + s2.split_whitespace().count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use std::f32;

    #[test]
    fn test_euclidian_distance() {
        let vec_one = vec![1.0, 2.0, 3.0];
        let vec_two = vec![2.0, 1.0, 4.0];

        let distance = euclidian_distance(vec_one, vec_two);

        assert!((distance - 1.7320508).abs() <= f32::EPSILON); 
    }

    #[test]
    fn test_cosine_similarity() {
        let vec_one = vec![2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0];
        let vec_two = vec![2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];

        let sim = cosine_similarity(vec_one, vec_two);

        assert_eq!(sim, 0.8215838);
    }

    #[test]
    fn test_jaccard_coeficient() {
        let vec_one = vec![2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0];
        let vec_two = vec![2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        
        let sim = jaccard_coeficient(vec_one, vec_two);

        assert_eq!(sim, 0.6923077);
    }

    #[test]
    fn test_matching_words_count() {
        let sen_one = String::from("le chat mange le oiseau.");
        let sen_two = String::from("le chien mange le chat.");

        let match_count = matching_words_count(&sen_one, &sen_two);

        assert_eq!(match_count, 3);
    }
    
    #[test]
    fn test_matching_words_frequency() {
        let sen_one = String::from("le chat mange le oiseau.");
        let sen_two = String::from("le chien mange le chat.");

        let match_count = matching_words_frequency(&sen_one, &sen_two);

        assert_eq!(match_count, 2);
    }

    #[test]
    fn test_matching_sim() {
        let sen_one = String::from("le chat mange.");
        let sen_two = String::from("le chien mange.");

        let sim = matching_sim(&sen_one, &sen_two, Matching::Count);

        assert_eq!(sim, 3/4);
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
