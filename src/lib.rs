#![crate_name = "narcissus"]

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

pub fn pearson_correlation(vec_one: Vec<f32>, vec_two: Vec<f32>) -> f32 {
    // Get the length of vectors
    let length = vec_one.len() as f32;

    // Summation over all attributes for both objects
    let sum_vec_one = vec_one.iter().fold(0.0, |sum, x| sum + x);
    let sum_vec_two = vec_two.iter().fold(0.0, |sum, x| sum + x);

    // Sum the squares
    let square_sum_one = vec_one.iter().map(|x| x * x).fold(0.0, |sum, x| sum + x);
    let square_sum_two = vec_two.iter().map(|x| x * x).fold(0.0, |sum, x| sum + x);

    // Add up the products
    let product = vec_one.iter().zip(vec_two.iter()).map(|(x, y)| x * y).fold(0.0, |sum, i| sum + i);

    // Calculate Pearson Correlation score
    // numerator = product - (sum_vec_one*sum_vec_two/len(vec_one))
    let numerator = product - (sum_vec_one * sum_vec_two / length);

    let denominator = ((square_sum_one - (sum_vec_one * sum_vec_one) / length) *
    (square_sum_two - (sum_vec_two * sum_vec_two) / length)).powf(0.5);

    if denominator == 0.0 {
        return 0.0
    } else {
        let result = numerator / denominator;

        // We need to return a distance measure
        if result < 0.0 {
            return result.abs();
        } else {
            return 1.0 - result;
        }
    }
}

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

pub fn matching_sim(s1: &String, s2: &String, matching: Matching) -> f32  {
    let mwc = match matching {
        Matching::Frequency => matching_words_frequency(&s1, &s2),
        Matching::Count => matching_words_count(&s1, &s2),
    } as f32;

    (2.0 * mwc) / ((s1.split_whitespace().count() as f32) + (s2.split_whitespace().count() as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    // use test::Bencher;
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
    fn test_pearson_correlation() {
        let vec_one = vec![2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0];
        let vec_two = vec![2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];

        let vec_one_bis = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let vec_three = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0];

        let vec_two_bis = vec![2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let vec_four = vec![2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

        let sim_one_two = pearson_correlation(vec_one, vec_two);
        let sim_one_three = pearson_correlation(vec_one_bis, vec_three);
        let sim_two_four = pearson_correlation(vec_two_bis, vec_four);

        assert_eq!(sim_one_two, 0.6464466);
        // assert_eq!(sim_one_three, 0.0);
        assert_eq!(sim_two_four, 1.0);
    }

    // #[test]
    // fn test_matching_words_count() {
    //     let sen_one = String::from("le chat mange le oiseau.");
    //     let sen_two = String::from("le chien mange le chat.");

    //     let match_count = matching_words_count(&sen_one, &sen_two);

    //     assert_eq!(match_count, 3);
    // }

    // #[test]
    // fn test_matching_words_frequency() {
    //     let sen_one = String::from("le chat mange le oiseau.");
    //     let sen_two = String::from("le chien mange le chat.");

    //     let match_count = matching_words_frequency(&sen_one, &sen_two);

    //     assert_eq!(match_count, 2);
    // }

    // #[test]
    // fn test_matching_sim() {
    //     let sen_one = String::from("von miller name game valuabl player 21/2 sack forc fumbl recov defens malik jackson touchdown");
    //     let sen_two = String::from("card tell coach told gari kubiak surpris man didn’t prompt");
    //     let sen_three = String::from("panther offens struggl bronco offens wasnt denver special teams—jordan norwood 61-yard punt return longest super bowl histori punter britton colquitt averag 459 yard kick constant chang field posit kicker brandon mcmanus nail field goal attempt");

    //     let sim = matching_sim(&sen_two, &sen_three, Matching::Frequency);

    //     assert_eq!(sim, 2.3);
    // }
}
