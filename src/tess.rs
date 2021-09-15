use super::*;

use std::ops::{Deref, DerefMut};
use std::ffi::CString;
use std::fmt::{Debug, Formatter};
use std::convert::TryInto;

pub use leptess::LepTess;
use leptess::capi;
use leptess::leptonica::Pix;

pub struct TessApi(LepTess);

impl Deref for TessApi {
    type Target = LepTess;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TessApi {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Debug for TessApi {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.as_raw_handle().fmt(f)
    }
}

impl TessApi {
    pub fn raw(&mut self) -> *mut capi::TessBaseAPI {
        self.0.as_raw_handle_mut().raw
    }

    pub fn new(data_path: Option<&str>, lang: &str) -> Result<TessApi, TesserocrError> {
        let lep_tess = LepTess::new(data_path, lang)?;
        Ok(Self(lep_tess))
    }

    pub fn ocr(&mut self, img: impl AsRef<[u8]>, width: u32, height: u32) -> Result<String, TesserocrError> {
        unsafe {
            capi::TessBaseAPISetPageSegMode(self.raw(), capi::TessPageSegMode_PSM_SINGLE_BLOCK);
        }

        self.set_image_from_buffer(width, height, img.as_ref())?;
        Ok(self.get_utf8_text()?)
    }

    pub fn set_variable(&mut self, key: &str, value: &str) -> Result<(), TesserocrError> {
        // maybe use a string pool to prevent leaking memory config variables
        let key = CString::new(key)?.into_raw();
        let value = CString::new(value)?.into_raw();
        unsafe {
            capi::TessBaseAPISetVariable(self.raw(), key, value);
        }

        Ok(())
    }

    /// Provide an image for Tesseract to recognize.
    ///
    /// This method clears all recognition results, and sets the rectangle to the full image, so it
    /// may be followed immediately by a `[Self::get_utf8_text]`, and it will automatically perform
    /// recognition.
    ///
    /// Unlike `[LepTess::set_image]` and `[LepTess::set_image_from_mem]`, this method takes an
    /// uncompressed image in RGB, RGBA or grayscale colors. This means in-memory images can be
    /// directly passed to this function. There is no need to save the image to a temporary file
    /// (`[LepTess::set_image]`) or compress it to an intermediate format
    /// (`[LepTess::set_image_from_mem]`).
    ///
    /// This function automatically determines the number color channels from the number of bytes of
    /// the given image. If the number of pixels (as calculated from `width * height`)
    /// in the image is equal to this number each pixel occupies exactly one byte.
    /// This implies each pixel has exactly one color channel and
    /// the input image is treated as a grayscale image. The same applies to three bytes and
    /// four bytes per pixel with RGB and RGBA images, respectively. Should none of these
    /// apply to the input image, the functions considers the input image to be invalid and
    /// will return an error.
    pub fn set_image_from_buffer(&mut self, width: u32, height: u32, img: &[u8]) -> Result<(), TesserocrError> {
        let op = |_| "Out of bounds";
        let width: i32 = width.try_into().map_err(op)?;
        let height: i32 = height.try_into().map_err(op)?;
        let img_len: i32 = img.len().try_into().map_err(op)?;
        let pixel_count = width * height;

        let (img_channel_count, pix_channel_count) = match img_len / pixel_count {
            3 => (3, 4), // store rgb images in rgba layout because of alignment
            c @ (1 | 4) => (c, c),
            _ => return Err("Supplied image buffer has invalid size".into())
        };

        let pix = unsafe {
            // SAFETY: We made sure that the arguments are valid:
            // - width and height must be i32
            // - bits_per_pixel must be one of 8, 16, 24 or 32
            // plus, pixCreate checks these constraints
            let bits_per_pixel = 8 * pix_channel_count;
            let pix = capi::pixCreate(width, height, bits_per_pixel);
            if pix.is_null() {
                return Err("Couldn't create the pix".into())
            }

            Pix { raw: pix }
        };

        let pix_data: &mut [u8] = unsafe {
            // SAFETY: Leptonica allocates the pix data buffer (does the null check)
            // and fills it with zeros => we can safely create a reference to this memory
            let data = capi::pixGetData(pix.raw).cast();
            let len = (pixel_count * pix_channel_count) as usize;
            std::slice::from_raw_parts_mut(data, len)
        };

        match img_channel_count {
            3 => { // rgb
                for (&img_rgb, pix_rgba) in chunks_exact::<_, 3>(img)
                    .zip(chunks_exact_mut::<_, 4>(pix_data)) {
                    let [r, g, b] = img_rgb;
                    let alpha = 255; // max alpha means fully opaque
                    *pix_rgba = [r, g, b, alpha];
                }
            }
            1 | 4 => pix_data.copy_from_slice(img),
            _ => return Err("Supplied image buffer has invalid size".into())
        }

        self.as_raw_handle_mut().set_image(&pix);

        Ok(())
    }
}

fn chunks_exact<T, const N: usize>(iter: &[T]) -> impl Iterator<Item=&[T; N]> {
    iter.chunks_exact(N)
        .map(|chunk|
            chunk.try_into()
                .unwrap_or_else(|_| panic!("chunks should have length of {}", N))
        )
}

fn chunks_exact_mut<T, const N: usize>(iter: &mut [T]) -> impl Iterator<Item=&mut [T; N]> {
    iter.chunks_exact_mut(N)
        .map(|chunk|
                 chunk.try_into()
                     .unwrap_or_else(|_| panic!("chunks should have length of {}", N))
        )
}
