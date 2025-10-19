//! Rust equivalent of <https://github.com/scipy/scipy/blob/71fb82e33c47a4358804578a6a16ed9129ef4049/scipy/stats/_ansari_swilk_statistics.pyx>

#![allow(clippy::cast_possible_wrap, dead_code)]

use std::cmp::Ordering;

use ndarray::{Array1, ArrayView1, ArrayViewMut1, s};

use crate::Float;

//=================================================================
// gscale (f32) and its helpers
//=================================================================

/// Rust translation of the gscale function.
///
/// Original docs:
/// Cython translation for the FORTRAN 77 code given in:
///
/// Dinneen, L. C. and Blakesley, B. C., "Algorithm AS 93: A Generator for the
/// Null Distribution of the Ansari-Bradley W Statistic", Applied Statistics,
/// 25(1), 1976, :doi:`10.2307/2346534`
fn gscale(test: i32, other: i32) -> (i32, Array1<f32>, i32) {
    let m = test.min(other);
    let n = test.max(other);
    let astart = ((test + 1) / 2) * (1 + (test / 2));
    let ll = (test * other) / 2 + 1;
    let ll_usize = ll as usize;

    let mut len1: i32;
    let mut len2: i32;
    let mut len3: i32 = 0; // Nonzero entry lengths

    let symm = (m + n) % 2 == 0;
    let odd = n % 2; // 0 or 1

    let mut a1 = Array1::<f32>::zeros(ll_usize);
    let mut a2 = Array1::<f32>::zeros(ll_usize);
    let mut a3 = Array1::<f32>::zeros(ll_usize);

    if m < 0 {
        return (0, Array1::<f32>::zeros(0), 2);
    }

    // Small cases
    if m == 0 {
        if ll_usize > 0 {
            a1[0] = 1.0;
        }
        return (astart, a1, 0);
    }

    if m == 1 {
        start1(a1.view_mut(), n);
        // `not (symm or (other > test))` is `!symm && !(other > test)` or `!symm && other <= test`
        if !symm && other <= test {
            if ll_usize > 0 {
                a1[0] = 1.0;
            }
            if ll_usize > 1 {
                a1[ll_usize - 1] = 2.0;
            }
        }
        return (astart, a1, 0);
    }

    if m == 2 {
        start2(a1.view_mut(), n);
        // `not (symm or (other > test))`
        if !symm && other <= test {
            for ind in 0..(ll_usize / 2) {
                a1.swap(ind, ll_usize - 1 - ind);
            }
        }
        return (astart, a1, 0);
    }

    // m > 2, Initialize for odd or even case
    let mut n2b1: i32;
    let mut n2b2: i32;
    let mut part_no: i32;
    let mut loop_m = 3;

    if odd == 1 {
        start1(a1.view_mut(), n);
        start2(a2.view_mut(), n - 1);
        len1 = 1 + (n / 2);
        len2 = n;
        n2b1 = 1;
        n2b2 = 2;
        part_no = 0;
    } else {
        start2(a1.view_mut(), n);
        start1(a2.view_mut(), n - 1);
        start2(a3.view_mut(), n - 2);
        len1 = n + 1;
        len2 = n / 2;
        len3 = n - 1;
        n2b1 = 2;
        n2b2 = 1;
        part_no = 1;
    }

    // loop_m can only increase hence "safe" while
    while loop_m <= m {
        if part_no == 0 {
            let l1out = frqadd(a1.view_mut(), a2.view(), len2, n2b1);
            len1 += n;
            len3 = imply(a1.view_mut(), l1out, len1, a3.view_mut(), loop_m);
            n2b1 += 1;
            loop_m += 1;
            part_no = 1;
        } else {
            let l2out = frqadd(a2.view_mut(), a3.view(), len3, n2b2);
            len2 += n - 1;
            // The return value is unused in the Cython code, assigned to `_`
            _ = imply(a2.view_mut(), l2out, len2, a3.view_mut(), loop_m);
            n2b2 += 1;
            loop_m += 1;
            part_no = 0;
        }
    }

    if !symm {
        let ks = i32::midpoint(m, 3) - 1;
        let ks_usize = ks as usize;
        let len2_usize = len2 as usize;
        for ind in 0..len2_usize {
            if ks_usize + ind < ll_usize {
                a1[ks_usize + ind] += a2[ind];
            }
        }
    }

    // reverse array
    if other > test {
        for ind in 0..(ll_usize / 2) {
            a1.swap(ind, ll_usize - 1 - ind);
        }
    }

    (astart, a1, 0)
}

/// Helper function for gscale function
fn start1(mut a: ArrayViewMut1<f32>, n: i32) {
    let lout = 1 + (n / 2);
    let lout_usize = lout as usize;

    if lout_usize > 0 {
        a.slice_mut(s![0..lout_usize]).fill(2.0);
    }

    if n % 2 == 0 && lout_usize > 0 {
        a[lout_usize - 1] = 1.0;
    }
}

/// Helper function for gscale function
fn start2(mut a: ArrayViewMut1<f32>, n: i32) {
    let odd = n % 2;
    let mut a_val = 1.0; // Cython `A`
    let mut b_val = 3.0; // Cython `B`
    let c_val = if odd == 1 { 2.0 } else { 0.0 }; // Cython `C`
    let ndo = (n + 2 + odd) / 2 - odd;

    for ind in 0..(ndo as usize) {
        a[ind] = a_val;
        a_val += b_val;
        b_val = 4.0 - b_val;
    }

    a_val = 1.0;
    b_val = 3.0;

    for ind in ((ndo as usize)..=((n - odd) as usize)).rev() {
        a[ind] = a_val + c_val;
        a_val += b_val;
        b_val = 4.0 - b_val;
    }

    if odd == 1 {
        let idx = (ndo * 2 - 1) as usize;
        if idx < a.len() {
            a[idx] = 2.0;
        }
    }
}

/// Helper function for gscale function
fn frqadd(mut a: ArrayViewMut1<f32>, b: ArrayView1<f32>, lenb: i32, offset: i32) -> i32 {
    let two = 2.0_f32;
    let lout = lenb + offset;
    let offset_usize = offset as usize;

    for ind in 0..(lenb as usize) {
        let a_idx = offset_usize + ind;
        if a_idx < a.len() && ind < b.len() {
            a[a_idx] += two * b[ind];
        }
    }

    lout
}

/// Helper function for gscale function
fn imply(
    mut a: ArrayViewMut1<f32>,
    curlen: i32,
    reslen: i32,
    mut b: ArrayViewMut1<f32>,
    offset: i32,
) -> i32 {
    let mut i2 = -offset;
    let mut j2 = reslen - offset;
    let j2min = (j2 + 1) / 2 - 1;
    let nextlenb = j2;
    let mut j1 = reslen - 1;

    j2 -= 1;

    for i1_i32 in 0..((reslen + 1) / 2) {
        let i1 = i1_i32 as usize;
        let summ;

        if i2 < 0 {
            summ = a[i1];
        } else {
            summ = a[i1] + b[i2 as usize];
            a[i1] = summ;
        }

        i2 += 1;
        if j2 >= j2min {
            let diff = if j1 > curlen - 1 { summ } else { summ - a[j1 as usize] };

            b[i1] = diff;
            b[j2 as usize] = diff;
            j2 -= 1;
        }

        a[j1 as usize] = summ;
        j1 -= 1;
    }

    nextlenb
}

/// Helper function for swilk.
///
/// Evaluates the tail area of the standardized normal curve from x to inf
/// if upper is True or from -inf to x if upper is False
fn alnorm<T: Float>(x: T, mut upper: bool) -> T {
    let t_from = |v: f64| T::from(v).unwrap();
    let ltone = 7.0;
    // let utzero = 18.66; // Original
    let utzero = 38.0; // Modified in Nov 2001
    let con = 1.28;

    let a1 = t_from(0.398_942_280_444);
    let a2 = t_from(0.399_903_438_504);
    let a3 = t_from(5.758_854_804_58);
    let a4 = t_from(29.821_355_780_8);
    let a5 = t_from(2.624_331_216_79);
    let a6 = t_from(48.695_993_069_2);
    let a7 = t_from(5.928_857_244_38);
    let b1 = t_from(0.398_942_280_385);
    let b2 = t_from(3.805_2_E-8);
    let b3 = t_from(1.000_006_153_02);
    let b4 = t_from(3.980_647_94_E-4);
    let b5 = t_from(1.986_153_813_64);
    let b6 = t_from(0.151_679_116_635);
    let b7 = t_from(5.293_303_249_26);
    let b8 = t_from(4.838_591_280_8);
    let b9 = t_from(15.150_897_245_1);
    let b10 = t_from(0.742_380_924_027);
    let b11 = t_from(30.789_933_034);
    let b12 = t_from(3.990_194_170_11);

    let mut z = x;

    if !matches!(z.partial_cmp(&t_from(0.0)), Some(Ordering::Greater)) {
        // Catches NaNs and <= 0
        upper = false;
        z = -z;
    }

    if !((z <= t_from(ltone)) || (upper && z <= t_from(utzero))) {
        return if upper { t_from(0.0) } else { t_from(1.0) };
    }

    let y = t_from(0.5) * z * z;

    let temp = if z <= t_from(con) {
        t_from(0.5) - z * (a1 - a2 * y / (y + a3 - a4 / (y + a5 + a6 / (y + a7))))
    } else {
        b1 * (-y).exp()
            / (z - b2
                + b3 / (z + b4 + b5 / (z - b6 + b7 / (z + b8 - b9 / (z + b10 + b11 / (z + b12))))))
    };

    if upper { temp } else { t_from(1.0) - temp }
}

/// Helper function for swilk. (PPND - Percent Point Normal Distribution)
fn ppnd<T: Float>(p: T) -> T {
    let t_from = |v: f64| T::from(v).unwrap();
    let a0 = t_from(2.506_628_238_84);
    let a1 = t_from(-18.615_000_625_29);
    let a2 = t_from(41.391_197_735_34);
    let a3 = t_from(-25.441_060_496_37);
    let b1 = t_from(-8.473_510_930_90);
    let b2 = t_from(23.083_367_437_43);
    let b3 = t_from(-21.062_241_018_26);
    let b4 = t_from(3.130_829_098_33);
    let c0 = t_from(-2.787_189_311_38);
    let c1 = t_from(-2.297_964_791_34);
    let c2 = t_from(4.850_141_271_35);
    let c3 = t_from(2.321_212_768_58);
    let d1 = t_from(3.543_889_247_62);
    let d2 = t_from(1.637_067_818_97);
    let split = t_from(0.42);
    let q = p - t_from(0.5);
    let mut temp;

    if q.abs() <= split {
        let r = q * q;
        temp = q * (((a3 * r + a2) * r + a1) * r + a0);
        temp /= (((b4 * r + b3) * r + b2) * r + b1) * r + t_from(1.0);

        return temp;
    }

    let mut r = p;
    if q > t_from(0.0) {
        r = t_from(1.0) - p;
    }

    if r > t_from(0.0) {
        r = (-r.ln()).sqrt();
    } else {
        return t_from(0.0);
    }

    temp = ((c3 * r + c2) * r + c1) * r + c0;
    temp /= (d2 * r + d1) * r + t_from(1.0);

    if q < t_from(0.0) { -temp } else { temp }
}

/// Helper function for swilk function that evaluates polynomials.
/// Coefficients are given as [c0, cn, cn-1, ..., c2, c1]
fn poly<T: Float>(c: &[f64], nord: i32, x: T) -> T {
    let t_from = |v: f64| T::from(v).unwrap();
    let nord_usize = nord as usize;
    let mut res = t_from(c[0]);

    if nord == 1 {
        return res;
    }

    let mut p = x * t_from(c[nord_usize - 1]);
    if nord == 2 {
        return res + p;
    }

    // Loop from nord-2 down to 1
    for ind in (1..=(nord_usize - 2)).rev() {
        p = (p + t_from(c[ind])) * x;
    }

    res += p;
    res
}

//=================================================================
// swilk (f64) and its helpers
//=================================================================

/// Calculates the Shapiro-Wilk W test and its significance level
///
/// This is a double precision Rust translation (with modifications) of the
/// FORTRAN 77 code given in:
///
/// Royston P., "Remark AS R94: A Remark on Algorithm AS 181: The W-test for
/// Normality", 1995, Applied Statistics, Vol. 44, :doi:`10.2307/2986146`
///
/// IFAULT error code details from the R94 paper:
/// - 0 for no fault
/// - 1 if N, N1 < 3
/// - 2 if N > 5000 (a non-fatal error)
/// - 3 if N2 < N/2, so insufficient storage for A
/// - 4 if N1 > N or (N1 < N and N < 20)
/// - 5 if the proportion censored (N-N1)/N > 0.8
/// - 6 if the data have zero range (if sorted on input)
///
/// For SciPy, n1 is never used, set to a positive number to enable
/// the functionality. Otherwise n1 = n is used.
#[allow(unused_assignments)]
pub(crate) fn swilk<T: Float>(
    x: ArrayView1<T>,
    mut a: ArrayViewMut1<T>,
    mut init: bool,
    n1_in: i32,
) -> (T, T, i32) {
    let n_usize = x.len();
    let n = n_usize as i32;
    let n2_usize = a.len();
    let mut _ind2: i32;
    let t_from = |v: f64| T::from(v).unwrap();

    let c1: [f64; 6] = [0.0, 0.221_157, -0.147_981, -0.207_119_E1, 0.443_468_5_E1, -0.270_605_6_E1];

    let c2: [f64; 6] =
        [0.0, 0.429_81_E-1, -0.293_762, -0.175_246_1_E1, 0.568_263_3_E1, -0.358_263_3_E1];

    let c3: [f64; 4] = [0.5440, -0.399_78, 0.250_54_E-1, -0.671_4_E-3];
    let c4: [f64; 4] = [0.138_22_E1, -0.778_57, 0.627_67_E-1, -0.203_22_E-2];
    let c5: [f64; 4] = [-0.158_61_E1, -0.310_82, -0.837_51_E-1, 0.389_15_E-2];
    let c6: [f64; 3] = [-0.4803, -0.826_76_E-1, 0.303_02_E-2];
    let c7: [f64; 2] = [0.164, 0.533];
    let c8: [f64; 2] = [0.1736, 0.315];
    let c9: [f64; 2] = [0.256, -0.635_E-2];
    let g: [f64; 2] = [-0.227_3_E1, 0.459];

    let z90 = t_from(0.128_16_E1);
    let z95 = t_from(0.164_49_E1);
    let z99 = t_from(0.232_63_E1);
    let zm = t_from(0.175_09_E1);
    let zss = t_from(0.562_68);
    let bf1 = t_from(0.8378);
    let xx90 = t_from(0.556);
    let xx95 = t_from(0.622);
    let sqrth = t_from(std::f64::consts::SQRT_2 / 2.0);
    let pi6 = t_from(6.0 / std::f64::consts::PI);
    let small = t_from(1e-19);
    let one = T::one();
    let zero = T::zero();

    let mut n1 = n1_in;

    if n1 < 0 {
        n1 = n;
    }

    let nn2: i32 = n / 2;
    let nn2_usize = nn2 as usize;

    if n2_usize < nn2_usize {
        return (one, one, 3); // IFAULT = 3
    }

    if n < 3 {
        return (one, one, 1); // IFAULT = 1
    }

    let an = t_from(f64::from(n));

    if !init {
        if n == 3 {
            a[0] = sqrth;
        } else {
            let an25 = an + t_from(0.25);
            let mut summ2 = zero;
            for ind1 in 0..nn2_usize {
                let temp = ppnd((t_from(ind1 as f64 + 1.0 - 0.375)) / an25);
                a[ind1] = temp;
                summ2 += temp.powi(2);
            }

            summ2 *= t_from(2.0);
            let ssumm2 = summ2.sqrt();
            let rsn = one / an.sqrt();
            let a1 = poly(&c1, 6, rsn) - (a[0] / ssumm2);

            let i1: usize;
            let fac: T;
            if n > 5 {
                i1 = 2;
                let a2 = -a[1] / ssumm2 + poly(&c2, 6, rsn);
                let num = summ2 - (t_from(2.0) * a[0].powi(2)) - (t_from(2.0) * a[1].powi(2));
                let den = one - (t_from(2.0) * a1.powi(2)) - (t_from(2.0) * a2.powi(2));
                fac = (num / den).sqrt();
                a[1] = a2;
            } else {
                i1 = 1;
                let num = summ2 - t_from(2.0) * a[0].powi(2);
                let den = one - t_from(2.0) * a1.powi(2);
                fac = (num / den).sqrt();
            }

            a[0] = a1;
            for ind1 in i1..nn2_usize {
                a[ind1] = a[ind1] * -one / fac;
            }
        }

        init = true;
    }

    if n1 < 3 {
        return (one, one, 1);
    }

    let ncens = n - n1;

    if ncens < 0 || (ncens > 0 && n < 20) {
        return (one, one, 4);
    }

    let delta = t_from(f64::from(ncens)) / an;
    if delta > t_from(0.8) {
        return (one, one, 5);
    }

    let range = x[(n1 - 1) as usize] - x[0];
    if range < small {
        return (one, one, 6);
    }

    let mut sx = x[0] / range;
    let mut sa = -a[0];
    let mut ind2 = n - 2;
    for ind1_i32 in 1..n1 {
        let ind1_usize = ind1_i32 as usize;
        let xi = x[ind1_usize] / range;
        sx += xi;

        if ind1_i32 != ind2 {
            let ind_min = ind1_i32.min(ind2) as usize;
            let sign = if ind1_i32 < ind2 { -one } else { one };
            sa += sign * a[ind_min];
        }

        ind2 -= 1;
    }

    let ifault = if n > 5000 { 2 } else { 0 };

    sa /= t_from(f64::from(n1));
    sx /= t_from(f64::from(n1));
    let mut ssa = zero;
    let mut ssx = zero;
    let mut sax = zero;
    ind2 = n - 1;

    for ind1_i32 in 0..n1 {
        let ind1_usize = ind1_i32 as usize;
        let asa = if ind1_i32 == ind2 {
            -sa
        } else {
            let ind_min = (ind1_i32.min(ind2)) as usize;
            let sign = if ind1_i32 < ind2 { -one } else { one };
            sign * a[ind_min] - sa
        };

        let xsx = x[ind1_usize] / range - sx;

        ssa += asa * asa;
        ssx += xsx * xsx;
        sax += asa * xsx;
        ind2 -= 1;
    }

    let ssassx = (ssa * ssx).sqrt();
    let w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
    let w = one - w1;

    if n == 3 {
        if w < t_from(0.75) {
            return (t_from(0.75), zero, ifault);
        }
        let pw = one - pi6 * w.sqrt().acos();
        return (w, pw, ifault);
    }

    let y = w1.ln();
    let xx = an.ln();
    let m: T;
    let s: T;

    if n <= 11 {
        let gamma = poly(&g, 2, an);
        if y >= gamma {
            return (w, small, ifault);
        }

        let y_transformed = -(gamma - y).ln();
        m = poly(&c3, 4, an);
        s = poly(&c4, 4, an).exp();

        let pw = alnorm((y_transformed - m) / s, true);
        return (w, pw, ifault);
    }

    m = poly(&c5, 4, xx);
    s = poly(&c6, 3, xx).exp();

    let mut final_m = m;
    let mut final_s = s;

    if ncens > 0 {
        let ld = -delta.ln();
        let bf = one + xx * bf1;
        let z90f = z90 + bf * poly(&c7, 2, xx90.powf(xx)).powf(ld);
        let z95f = z95 + bf * poly(&c8, 2, xx95.powf(xx)).powf(ld);
        let z99f = z99 + bf * poly(&c9, 2, xx).powf(ld);
        let zfm = (z90f + z95f + z99f) / t_from(3.0);
        let zsd_num = z90 * (z90f - zfm) + z95 * (z95f - zfm) + z99 * (z99f - zfm);
        let zsd = zsd_num / zss;
        let zbar = zfm - zsd * zm;
        final_m = m + zbar * s;
        final_s = s * zsd;
    }

    let pw = alnorm((y - final_m) / final_s, true);
    (w, pw, ifault)
}
