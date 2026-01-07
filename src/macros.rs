#[macro_export]
macro_rules! iter_if_parallel {
    ($collection:expr) => {{
        #[cfg(feature = "parallel")]
        let iter = $collection.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = $collection.iter();
        iter
    }};
}

#[macro_export]
macro_rules! sort_if_parallel {
    ($collection:expr, $compare:expr) => {
        #[cfg(feature = "parallel")]
        rayon::prelude::ParallelSliceMut::par_sort_unstable_by($collection, $compare);
        #[cfg(not(feature = "parallel"))]
        $collection.sort_unstable_by($compare);
    };
}
