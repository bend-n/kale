use std::any::Any;
use tinyvec::TinyVec;
struct Array {
    shape: Shape,
    data: Vec<Box<dyn Any>>,
}
struct Shape {
    dims: TinyVec<[usize; 3]>,
}

impl Shape {
    pub fn scalar() -> Self {
        Shape {
            dims: TinyVec::new(),
        }
    }
}
