from mmdet3d.models import DETECTORS

from .bevdet_occ import BEVStereo4DOCC


@DETECTORS.register_module()
class BEVStereo4DOCCCompat(BEVStereo4DOCC):
    """
    Compatibility wrapper for the official FlashOcc M3 model.

    It preserves the complete BEVStereo4DOCC architecture and only
    adapts occupancy decoding during inference:
        - prefer get_occ_gpu() when the head provides it;
        - otherwise fall back to the official get_occ().
    """

    def simple_test_occ(self, img_feats, img_metas=None):
        outs = self.occ_head(img_feats)

        if hasattr(self.occ_head, 'get_occ_gpu'):
            occ_preds = self.occ_head.get_occ_gpu(
                outs,
                img_metas
            )
        else:
            occ_preds = self.occ_head.get_occ(
                outs,
                img_metas
            )

        return occ_preds
