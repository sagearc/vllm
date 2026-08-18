// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Image-modality preparation: batch preprocessing and per-item feature
//! build.

use std::sync::Arc;

use llm_multimodal::{
    ImageFrame, Modality, MultiModalProcessorMetadata, PreprocessedEncoderInputs,
};
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
        model_dtype: Option<ModelDtype>,
    ) -> Result<PreparedMedia> {
        let support = self.image.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Image.to_string(),
        })?;
        let hashes = frames.iter().map(|frame| frame.hash.clone()).collect();
        let (replacements, items) = match model_dtype {
            Some(model_dtype) => {
                let preprocessed = self.preprocess_images(support, &frames).await?;
                let replacements =
                    support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
                let items = item::build_batched_items(
                    &support.spec,
                    preprocessed,
                    hashes,
                    uuids,
                    model_dtype,
                )?;
                (replacements, items)
            }
            None => {
                let metadata = self.preprocess_image_metadata(support, &frames).await?;
                let replacements =
                    support.spec.prompt_replacements_for_metadata(&self.context, &metadata)?;
                let items = item::build_items_without_data(hashes, uuids)?;
                (replacements, items)
            }
        };
        if replacements.len() != frames.len() {
            bail_multimodal!(
                "number of image prompt replacements {} does not match number of images {}",
                replacements.len(),
                frames.len()
            );
        }
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

    /// Compute image prompt metadata without constructing encoder tensors.
    async fn preprocess_image_metadata(
        &self,
        support: &ModalitySupport,
        image_frames: &[Arc<ImageFrame>],
    ) -> Result<MultiModalProcessorMetadata> {
        let config = support.config.clone();
        let processor = support.processor;
        let images = image_frames.iter().map(|frame| frame.data().clone()).collect::<Vec<_>>();

        tokio::task::spawn_blocking(move || {
            processor.preprocess_metadata(&images, &config)?.ok_or_else(|| {
                multimodal!("model does not support metadata-only image preprocessing")
            })
        })
        .await
        .map_err(|error| multimodal!("image metadata preprocessing task failed: {error}"))?
    }
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;
    use llm_multimodal::{
        MediaContentPart, PreProcessorConfig, TransformError, VideoClip, VideoSource,
        VisionPreProcessor,
    };

    use super::super::tests::qwen3_vl_info;
    use super::*;

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
            panic!("render-only image path called full preprocessing")
        }

        fn preprocess_metadata(
            &self,
            images: &[DynamicImage],
            _config: &PreProcessorConfig,
        ) -> std::result::Result<Option<MultiModalProcessorMetadata>, TransformError> {
            Ok(Some(MultiModalProcessorMetadata::new(
                vec![1; images.len()],
                images.iter().map(|image| (image.width(), image.height())).collect(),
            )))
        }

        fn preprocess_video_metadata(
            &self,
            frames: &[DynamicImage],
            _config: &PreProcessorConfig,
        ) -> std::result::Result<Option<MultiModalProcessorMetadata>, TransformError> {
            let first = frames.first().ok_or(TransformError::EmptyBatch)?;
            Ok(Some(MultiModalProcessorMetadata::new(
                vec![1],
                vec![(first.width(), first.height())],
            )))
        }

        fn calculate_num_tokens(
            &self,
            width: u32,
            height: u32,
            config: &PreProcessorConfig,
        ) -> usize {
            let _ = (width, height, config);
            1
        }

        fn model_name(&self) -> &'static str {
            "metadata-only-test"
        }
    }

    static METADATA_ONLY_PROCESSOR: MetadataOnlyProcessor = MetadataOnlyProcessor;

    #[tokio::test]
    async fn render_only_vision_never_calls_full_preprocessing() {
        let mut info = qwen3_vl_info();
        info.image.as_mut().unwrap().processor = &METADATA_ONLY_PROCESSOR;
        info.video.as_mut().unwrap().processor = &METADATA_ONLY_PROCESSOR;
        let fetched = info
            .fetch_media(vec![MediaContentPart::ImageUrl {
                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=".to_string(),
                detail: None,
                uuid: Some("image-1".to_string()),
            }])
            .await
            .unwrap();

        let rendered =
            info.prepare_images(fetched.images, fetched.image_uuids, None).await.unwrap();

        let video = Arc::new(VideoClip::new(
            vec![
                DynamicImage::new_rgb8(32, 32),
                DynamicImage::new_rgb8(32, 32),
            ],
            bytes::Bytes::from_static(b"video"),
            VideoSource::InlineBytes,
            "video-hash".to_string(),
        ));
        let rendered_video = info
            .prepare_videos(vec![video], vec![Some("video-1".to_string())], None)
            .await
            .unwrap();

        assert!(rendered.items[0].data.is_none());
        assert!(rendered_video.items[0].data.is_none());
    }
}
