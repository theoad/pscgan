import datasets.base as base
import datasets.lightning as lightning

factory = {
    'base': lightning.BaseDatamoduleLightning,
    'lightning-resize': lightning.ResizeDataset,
    'lightning-crop': lightning.RandomCropDataset,
    'lightning-crop-resize': lightning.CenterCropResizeDataset,
    'lightning-no-transform': lightning.NoTransormDataset,
}
