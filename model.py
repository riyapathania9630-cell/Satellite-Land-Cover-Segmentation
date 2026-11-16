"""
model.py
========
Complete U-Net architecture implementation using TensorFlow/Keras, as
described in Ronneberger et al., 2015 ("U-Net: Convolutional Networks
for Biomedical Image Segmentation"), adapted for multi-class land cover
segmentation.

The network follows the classic encoder-decoder shape with skip
connections between corresponding encoder/decoder resolution levels.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

import config


def conv_block(inputs, num_filters: int, dropout_rate: float = 0.0, name_prefix: str = ""):
    """
    Two consecutive Conv2D -> BatchNorm -> ReLU blocks, the fundamental
    building block of U-Net used at every resolution level.

    Args:
        inputs: Input tensor.
        num_filters: Number of convolution filters.
        dropout_rate: Optional dropout applied after the block.
        name_prefix: Prefix for layer names (helps readability in
            model.summary() and TensorBoard graphs).

    Returns:
        Output tensor after two conv-bn-relu blocks.
    """
    x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal",
                       name=f"{name_prefix}_conv1")(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal",
                       name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)
    return x


def encoder_block(inputs, num_filters: int, dropout_rate: float = 0.0, name_prefix: str = ""):
    """
    One encoder (downsampling) stage: a conv_block followed by 2x2 max
    pooling. Returns both the pre-pooled feature map (for the skip
    connection) and the pooled output (fed to the next encoder stage).
    """
    skip = conv_block(inputs, num_filters, dropout_rate, name_prefix)
    pooled = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name_prefix}_pool")(skip)
    return skip, pooled


def decoder_block(inputs, skip_features, num_filters: int, dropout_rate: float = 0.0, name_prefix: str = ""):
    """
    One decoder (upsampling) stage: transposed convolution to double the
    spatial resolution, concatenation with the matching encoder skip
    connection, then a conv_block to refine features.
    """
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding="same",
                                name=f"{name_prefix}_upconv")(inputs)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip_features])
    x = conv_block(x, num_filters, dropout_rate, name_prefix)
    return x


def build_unet(input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
                num_classes: int = config.NUM_CLASSES,
                use_categorical: bool = config.USE_CATEGORICAL,
                base_filters: int = 64) -> Model:
    """
    Build the complete U-Net model.

    Architecture:
        Encoder: 4 downsampling stages (base_filters, x2, x4, x8)
        Bottleneck: base_filters x16
        Decoder: 4 upsampling stages, mirroring the encoder with skip
                 connections
        Output: 1x1 conv with softmax (multi-class) or sigmoid (binary)

    Args:
        input_shape: (H, W, C) shape of input images.
        num_classes: Number of segmentation classes.
        use_categorical: If True, output is a softmax over num_classes
            channels. If False, output is a single sigmoid channel for
            binary segmentation.
        base_filters: Number of filters in the first encoder stage;
            doubles at every subsequent stage (standard U-Net design).

    Returns:
        An uncompiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ---------------- Encoder ----------------
    skip1, pool1 = encoder_block(inputs, base_filters, 0.1, "enc1")
    skip2, pool2 = encoder_block(pool1, base_filters * 2, 0.1, "enc2")
    skip3, pool3 = encoder_block(pool2, base_filters * 4, 0.2, "enc3")
    skip4, pool4 = encoder_block(pool3, base_filters * 8, 0.2, "enc4")

    # ---------------- Bottleneck ----------------
    bottleneck = conv_block(pool4, base_filters * 16, 0.3, "bottleneck")

    # ---------------- Decoder ----------------
    d1 = decoder_block(bottleneck, skip4, base_filters * 8, 0.2, "dec1")
    d2 = decoder_block(d1, skip3, base_filters * 4, 0.2, "dec2")
    d3 = decoder_block(d2, skip2, base_filters * 2, 0.1, "dec3")
    d4 = decoder_block(d3, skip1, base_filters, 0.1, "dec4")

    # ---------------- Output ----------------
    if use_categorical:
        outputs = layers.Conv2D(num_classes, 1, activation="softmax", name="output_mask")(d4)
    else:
        outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output_mask")(d4)

    model = Model(inputs, outputs, name="UNet_LandCover_Segmentation")
    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and print its summary + param count.
    net = build_unet()
    net.summary()
    print(f"\nTotal trainable parameters: {net.count_params():,}")
