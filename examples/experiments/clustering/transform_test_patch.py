from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
import os.path
import numpy as np


class AddRandomTask(EOTask):

    def execute(self, eopatch):
        print(eopatch)
        t, h, w, _ = eopatch.data['ndvi'].shape

        rand1 = np.random.rand(h, w, 1)
        rand2 = np.random.rand(h, w, 1)
        mask = np.zeros((5, 20))
        mask = np.concatenate((mask, np.ones((10, 20))))
        mask = np.concatenate((mask, np.zeros((5, 20))))

        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'feature1', rand1)
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'feature2', rand2)
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'mask', mask[..., np.newaxis])
        print(eopatch)
        return eopatch


patch_location = './eopatch/'
load = LoadFromDisk(patch_location)

save_path_location = './eopatch/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

extra_param = {
    load: {'eopatch_folder': 'TestPatch'},
    save: {'eopatch_folder': 'TestPatchNew'}
}

workflow = LinearWorkflow(
    load,
    AddRandomTask(),
    save
)

workflow.execute(extra_param)
