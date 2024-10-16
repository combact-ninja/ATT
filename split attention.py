




import tensorflow as tf


def split_attention_single_modality(input_tensor, block_channel, reduction_factor=4, lowest_atten=0.0):
    """
    TensorFlow implementation of split attention for a single modality.

    Args:
        input_tensor: A 4D tensor (batch_size, num_channels, height, width).
        block_channel: The channel number of each block.
        reduction_factor: Reduction factor for the intermediate attention calculation.
        lowest_atten: Minimum attention value.

    Returns:
        Tensor after applying split attention.
    """
    # Get the input tensor shape
    batch_size, num_channels, height, width = tf.shape(input_tensor)

    # Calculate number of blocks
    num_blocks = num_channels // block_channel

    # Reshape the input to [batch_size, num_blocks, block_channel, height, width]
    reshaped_x = tf.reshape(input_tensor, (batch_size, num_blocks, block_channel, height, width))

    # Global average pooling on each block (i.e., pooling across spatial dimensions)
    gap = tf.reduce_mean(reshaped_x, axis=[3, 4])  # Pool across height and width

    # Linear projection to reduced dimensions
    reduced_channels = block_channel // reduction_factor
    reduced_features = tf.keras.layers.Dense(reduced_channels)(gap)
    reduced_features = tf.keras.layers.ReLU()(reduced_features)

    # Attention calculation (Softmax on reduced features)
    attention_logits = tf.keras.layers.Dense(block_channel)(reduced_features)
    attention_values = tf.nn.softmax(attention_logits, axis=-1)

    # Apply lowest attention scaling
    attention_values = lowest_atten + attention_values * (1.0 - lowest_atten)

    # Reshape the attention values to match input dimensions
    attention_values = tf.reshape(attention_values, (batch_size, num_blocks, block_channel, 1, 1))

    # Apply attention to the reshaped input
    attended_x = reshaped_x * attention_values

    # Reshape back to original dimensions
    output_x = tf.reshape(attended_x, (batch_size, num_channels, height, width))

    return output_x


# # Sample input for a single modality (batch_size, channels, height, width)
# input_tensor = tf.random.normal([4, 32, 64, 64])
#
# # Apply the split attention function
# output = split_attention_single_modality(input_tensor, block_channel=8)
#
# print("Output shape:", output.shape)
