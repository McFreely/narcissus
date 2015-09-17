pub mod similarity {
    pub fn euclidian_distance(vec_one: Vec<f32>, vec_two: Vec<f32>) -> f32 {
        let mut result: f32 = 0.0;

        for (i, j) in vec_one.iter().zip(vec_two) {
            result += (i - j).abs().powf(2.0);
        }

        result.sqrt()
    }

    pub fn cosine_similarity(vec_one: Vec<f32>, vec_two: Vec<f32>) -> f32 {
        let dot_product = dot_product(&vec_one, &vec_two);
        let magnitude = magnitude(&vec_one, &vec_two); 

        println!("{:?}", dot_product);
        dot_product / magnitude
    }

    // pub fn jaccard_coeficient<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
    // }

    // pub fn pearson_correlation<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
    // }
    
    // pub fn kullback_liebler_divergence<T>(vec_one: Vec<T>, vec_two: Vec<T>) -> f32 {
    // }

    fn dot_product(vec_one: &Vec<f32>, vec_two: &Vec<f32>) -> f32 {
        let mut result: f32 = 0.0;

        for (i, j) in vec_one.iter().zip(vec_two) {
            result += i * j
        }

        result
    }

    fn magnitude(vec_one: &Vec<f32>, vec_two: &Vec<f32>) -> f32 {
        // The magnitude of a vector is the sqrt of its own dotproduct
        let dot_product_vector_one = dot_product(vec_one, vec_one);
        let dot_product_vector_two = dot_product(vec_two, vec_two);

        dot_product_vector_one.sqrt() * dot_product_vector_two.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::similarity::euclidian_distance;
    use super::similarity::cosine_similarity as cosine;
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
}
