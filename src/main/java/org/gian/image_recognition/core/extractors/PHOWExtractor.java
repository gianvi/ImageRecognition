package org.gian.image_recognition.core.extractors;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;


/**
 * Extracts feature vectors using the PHOW technique.
 * <p>
 * Adapted from: http://www.openimaj.org/tutorial/classification101.html
 */
public class PHOWExtractor implements FeatureExtractor<DoubleFV, ImageProvider<FImage>> {
    AbstractDenseSIFT<FImage> sift;
    HardAssigner<byte[], float[], IntFloatPair> assigner;

    public PHOWExtractor(AbstractDenseSIFT<FImage> sift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
        this.sift = sift;
        this.assigner = assigner;
    }

    public DoubleFV extractFeature(ImageProvider<FImage> provider) {
        FImage image = provider.getImage();
        sift.analyseImage(image);

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(bovw, 1, 1);

        return spatial.aggregate(sift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}
