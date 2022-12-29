import torchvision.transforms as transforms

import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import Pipeline, pipeline_def, fn, types

INPUT_SIZE = 224

random_perspective_transform = transforms.Compose([
                              transforms.RandomPerspective(p=0.5),
                              ])
def perspective(t):
    return random_perspective_transform(t)

def mux(condition, true_case, false_case):
    neg_condition = condition ^ True
    return condition * true_case + neg_condition * false_case

def create_pipeline_perspective(batch_size, num_threads, data_dir):

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=num_threads, exec_async=False, exec_pipelined=False)
    def _create_pipeline(data_dir):
        inputs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=True,
            prefetch_queue_depth=2,
            name="Reader",
        )

        output_layout = "CHW"
        
        images = fn.decoders.image(inputs, device = "cpu")

        images = fn.resize(
            images,
            dtype=types.UINT8,
            resize_x=INPUT_SIZE,
            resize_y=INPUT_SIZE,
            device="cpu"
        )

        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, vertical = flip_coin, device = "cpu") # Vertical Flip

        contrast_jitter = fn.random.uniform(range=[0.5, 1.5])
        saturation_jitter = fn.random.uniform(range=[0.5, 1.5])
        hue_jitter = fn.random.uniform(range=[0.5, 1.5])

        images_jittered = fn.color_twist(
            images, 
            contrast=contrast_jitter, 
            saturation=saturation_jitter,
            hue=hue_jitter
        )
        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images = mux(condition, images_jittered, images) # Color Jitter

        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images_gray = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY)
        images = mux(condition, images_gray, images) # Gray Scale

        images = fn.transpose(images, output_layout=output_layout) # Transform from HWC -> CHW

        images = dalitorch.fn.torch_python_function(images, function=perspective) # Random Perspective

        images = fn.normalize(images, dtype=types.FLOAT)

        labels = fn.squeeze(labels, axes=[0])

        return images.gpu(), labels.gpu()
    return _create_pipeline(data_dir)

def create_pipeline_no_perspective(batch_size, num_threads, data_dir):

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=num_threads)
    def _create_pipeline(data_dir):
        inputs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=True,
            prefetch_queue_depth=2,
            name="Reader",
        )

        output_layout = "CHW"
        
        images = fn.decoders.image(inputs, device = "mixed")

        images = fn.resize(
            images,
            dtype=types.UINT8,
            resize_x=INPUT_SIZE,
            resize_y=INPUT_SIZE,
            device="gpu"
        )

        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, vertical = flip_coin, device = "gpu") # Vertical Flip

        contrast_jitter = fn.random.uniform(range=[0.5, 1.5])
        saturation_jitter = fn.random.uniform(range=[0.5, 1.5])
        hue_jitter = fn.random.uniform(range=[0.5, 1.5])

        images_jittered = fn.color_twist(
            images, 
            contrast=contrast_jitter, 
            saturation=saturation_jitter,
            hue=hue_jitter
        )
        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images = mux(condition, images_jittered, images) # Color Jitter

        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images_gray = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY)
        images = mux(condition, images_gray, images) # Gray Scale

        images = fn.transpose(images, output_layout=output_layout) # Transform from HWC -> CHW

        images = fn.normalize(images, dtype=types.FLOAT)

        labels = fn.squeeze(labels, axes=[0])

        return images.gpu(), labels.gpu()
    return _create_pipeline(data_dir)