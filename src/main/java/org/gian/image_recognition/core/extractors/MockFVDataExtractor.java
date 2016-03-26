package org.gian.image_recognition.core.extractors;

import org.gian.image_recognition.core.data.FVData;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;


/**
 * Unlike most {@link FeatureExtractor} classes, this class simply retrieves the pre-calculated
 * feature vector packaged in the class. As such, this extractor class does not truly extract any
 * features.
 */
public class MockFVDataExtractor implements FeatureExtractor<DoubleFV, FVData> {
    @Override
    public DoubleFV extractFeature(FVData imageData) {
        return imageData.getFeatureVector();
    }
}
