package org.gian.image_recognition.core.data;


import org.openimaj.feature.DoubleFV;


public interface FVData extends ImageData {
    DoubleFV getFeatureVector();
}
