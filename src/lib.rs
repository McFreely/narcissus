pub mod similarity {
    
    pub fn euclidian_distance(vec_one: Vec<f32>, vec_two: Vec<f32>) -> f32 {
        let mut result: f32 = 0.0;
        let len = vec_one.len();

        for i in 0..len {
            let value_one = vec_one.get(i).unwrap();
            let value_two = vec_two.get(i).unwrap();
            result += (value_one - value_two).abs().powf(2.0);
        }

        result.sqrt()
    }

    // pub fn cosine_similarity<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
    // }

    // pub fn jaccard_coeficient<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
    // }

    // pub fn pearson_correlation<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
    // }
    
    // pub fn kullback_liebler_divergence<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
    // }
}

#[cfg(test)]
mod tests {
    use super::similarity::euclidian_distance;
    use std::f32;

    #[test]
    fn euclidian_distance_works() {
        let vec_one = vec![1.0, 2.0, 3.0];
        let vec_two = vec![2.0, 1.0, 4.0];

        let distance = euclidian_distance(vec_one, vec_two);

        assert!((distance - 1.7320508).abs() <= f32::EPSILON); 
    }
}
