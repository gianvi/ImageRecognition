package org.gian.image_recognition.core.data;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;

public interface KPData extends ImageData {
    LocalFeatureList<ByteDSIFTKeypoint> getKeypoints();
}
