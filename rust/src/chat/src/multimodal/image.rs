// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Image-modality preparation: batch preprocessing and per-item feature
//! build.

use std::sync::Arc;

use llm_multimodal::{ImageFrame, Modality, PreprocessedEncoderInputs};
use ndarray::Array1;
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::{ModalitySupport, MultimodalModelInfo, PreparedMedia, item};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary image encoder input.
pub(super) const IMAGE_PRIMARY_KEY: &str = "pixel_values";

impl MultimodalModelInfo {
    /// Preprocess all fetched image frames as one batch and build per-item
    /// features.
    pub(super) async fn prepare_images(
        &self,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
        model_dtype: ModelDtype,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;
        let preprocessed = self.preprocess_images(support, &frames).await?;
        let replacements = support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }
        let hashes = frames.iter().map(|frame| frame.hash.clone()).collect();
        let items =
            item::build_batched_items(&support.spec, preprocessed, hashes, uuids, model_dtype)?;

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    pub(super) fn prepare_image_metadata(
        &self,
        frames: Vec<Arc<ImageFrame>>,
        uuids: Vec<Option<String>>,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;
        let item_sizes = frames
            .iter()
            .map(|frame| (frame.data().width(), frame.data().height()))
            .collect::<Vec<_>>();
        let feature_token_counts = item_sizes
            .iter()
            .map(|&(width, height)| {
                support.processor.calculate_num_tokens(width, height, &support.config)
            })
            .collect();
        let metadata = PreprocessedEncoderInputs::new(
            Array1::<f32>::zeros(0),
            feature_token_counts,
            item_sizes,
        );
        let replacements = support.spec.prompt_replacements_for(&self.context, &metadata)?;
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }
        let hashes = frames.iter().map(|frame| frame.hash.clone()).collect();
        let items = item::build_items_without_data(hashes, uuids)?;

        Ok(PreparedMedia {
            modality: Modality::Image,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Preprocess fetched image frames with the model's resolved vision
    /// processor.
    ///
    /// The processor work is CPU-heavy relative to request wiring, so it runs
    /// in a blocking task and returns owned tensors ready for wire
    /// conversion.
    async fn preprocess_images(
        &self,
        support: &ModalitySupport,
        image_frames: &[Arc<ImageFrame>],
    ) -> Result<PreprocessedEncoderInputs> {
        let config = support.config.clone();
        let processor = support.processor;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        // TODO: is it still necessary given that we've already in a dedicated runtime?
        tokio::task::spawn_blocking(move || Ok(processor.preprocess(&images, &config)?))
            .await
            .map_err(|error| multimodal!("image preprocessing task failed: {error}"))?
    }
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;
    use llm_multimodal::{
        MediaContentPart, PreProcessorConfig, TransformError, VisionPreProcessor,
    };

    use super::super::tests::qwen3_vl_info;
    use super::*;

    const IMAGE_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

    struct MetadataOnlyProcessor;

    impl VisionPreProcessor for MetadataOnlyProcessor {
        fn default_mean(&self) -> [f64; 3] {
            [0.0; 3]
        }

        fn default_std(&self) -> [f64; 3] {
            [1.0; 3]
        }

        fn preprocess(
            &self,
            _images: &[DynamicImage],
            _config: &PreProcessorConfig,
        ) -> std::result::Result<PreprocessedEncoderInputs, TransformError> {
            panic!("render metadata invoked full image preprocessing")
        }

        fn calculate_num_tokens(
            &self,
            _width: u32,
            _height: u32,
            _config: &PreProcessorConfig,
        ) -> usize {
            4
        }

        fn model_name(&self) -> &'static str {
            "metadata-only-test"
        }
    }

    static METADATA_ONLY_PROCESSOR: MetadataOnlyProcessor = MetadataOnlyProcessor;

    async fn fetched_image(info: &MultimodalModelInfo) -> super::super::FetchedMedia {
        info.fetch_media(vec![MediaContentPart::ImageUrl {
            url: IMAGE_URL.to_string(),
            detail: None,
            uuid: Some("image-1".to_string()),
        }])
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn render_metadata_does_not_invoke_full_preprocessing() {
        let mut info = qwen3_vl_info();
        info.image.as_mut().unwrap().processor = &METADATA_ONLY_PROCESSOR;
        let fetched = fetched_image(&info).await;

        let prepared = info.prepare_image_metadata(fetched.images, fetched.image_uuids).unwrap();

        assert_eq!(prepared.replacements[0].tokens.len(), 4);
        assert!(prepared.items[0].data.is_none());
        assert_eq!(prepared.items[0].uuid.as_deref(), Some("image-1"));
    }

    #[tokio::test]
    async fn render_metadata_matches_inference_placeholder_tokens() {
        let info = qwen3_vl_info();
        let fetched = fetched_image(&info).await;
        let rendered = info
            .prepare_image_metadata(fetched.images.clone(), fetched.image_uuids.clone())
            .unwrap();
        let inferred = info
            .prepare_images(fetched.images, fetched.image_uuids, ModelDtype::Float32)
            .await
            .unwrap();

        assert_eq!(
            rendered.replacements[0].tokens,
            inferred.replacements[0].tokens
        );
        assert!(rendered.items[0].data.is_none());
        assert!(inferred.items[0].data.is_some());
    }
}
