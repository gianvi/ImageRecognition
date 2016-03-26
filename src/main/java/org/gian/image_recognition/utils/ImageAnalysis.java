package org.gian.image_recognition.utils;

import de.bwaldvogel.liblinear.SolverType;
import org.gian.image_recognition.core.data.FVData;
import org.gian.image_recognition.core.data.KPData;
import org.gian.image_recognition.core.data.LocalFVData;
import org.gian.image_recognition.core.data.LocalImageData;
import org.gian.image_recognition.core.extractors.CFIIFExtractor;
import org.gian.image_recognition.core.extractors.MockFVDataExtractor;
import org.gian.image_recognition.core.extractors.PHOWExtractor;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Contains various methods to help with image analysis
 */
public class ImageAnalysis {

    //richiamato dalla fase 3: ottieni un num fissato di centroidi dall'insieme di tutti KPS....CREA L'ASSIGNER
    public static ByteCentroidsResult trainQuantiser(List<KPData> keypoints, double percentage) {

        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        for (KPData kpdata : keypoints) {
            allkeys.add(kpdata.getKeypoints());

            System.out.println("L'immagine " + kpdata.getPath().substring(kpdata.getPath().length() - 15) + " contiene:   " + kpdata.getKeypoints().size() + " [" + kpdata.getKeypoints().vecLength() + "]");
        }


        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        int clusterNum = (int) (percentage / 100 * datasource.numRows());
        System.out.println("Universo U dei kps: " + datasource.numRows() + " clusters(" + percentage + " / " + 100 + ")*" + datasource.numRows() + "=" + clusterNum);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(clusterNum);


        return km.cluster(datasource);
    }


    //richiamato dalla fase 4: si estraggono le features del modello PHOW...per ogni immagine(sift, centroids) ed extract feature

    //richiamato anche dalla fase 5.2: per il test evaluation
    public static DoubleFV extractFeatures(HardAssigner<byte[], float[], IntFloatPair> assigner, ImageProvider image) {

        PHOWExtractor myPhowExtractor = new PHOWExtractor(getSift(), assigner);
        DoubleFV featVector = myPhowExtractor.extractFeature(image);
        return featVector;
    }

    public static void extractMyFeatures(HardAssigner<byte[], float[], IntFloatPair> assigner, List<KPData> corpus) {

        CFIIFExtractor myCFIIFExtractor = new CFIIFExtractor(getSift(), assigner, corpus);

        //DoubleFV featVector =

        myCFIIFExtractor.extractMyFeature(corpus);

        //return featVector;
    }


    //richiamato nella fase 5.1: ritorna il modello per il knn trainato con le PHOW feature e il MOCKExtractor!!
    public static LiblinearAnnotator<FVData, String> trainAnnotator(List<FVData> features) {
        BasicGroupedDataset<FVData> trainingDataset = ImageAnalysis.groupFeatures(features);
        MockFVDataExtractor extractor = new MockFVDataExtractor();
        LiblinearAnnotator<FVData, String> ann = new LiblinearAnnotator<FVData, String>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(trainingDataset);

        return ann;
    }

    public static BasicGroupedDataset<FVData> prepareTestData(String path, HardAssigner<byte[], float[], IntFloatPair> assigner, int numTestImagesPerClass) throws IOException {
        return prepareTestData(path, assigner, numTestImagesPerClass, 0);
    }


    public static BasicGroupedDataset<FVData> prepareTestData(String path, HardAssigner<byte[], float[], IntFloatPair> assigner, int numTestImagesPerClass, int start) throws IOException {
        File imageRoot = new File(path);

        List<FVData> testData = new ArrayList<FVData>();
        for (File folder : Arrays.asList(imageRoot.listFiles())) {
            if (folder.isDirectory()) {
                Iterable<File> files = Arrays.asList(folder.listFiles()).subList(start, start + numTestImagesPerClass);
                for (File file : files) {
                    System.out.println("Generating test image features");

                    DoubleFV fv = ImageAnalysis.extractFeatures(assigner, ImageUtilities.readF(file));

                    testData.add(new LocalFVData(file.getAbsolutePath(), folder.getName(), fv));
                }
            }
        }

        LocalImageData.writeList("tmp/5testdata.txt", testData);


        return ImageAnalysis.groupFeatures(testData);
    }


    //richiamato nella fase 5.1: sia nel train

    //richiamato in 5.2: nel test
    public static BasicGroupedDataset<FVData> groupFeatures(List<FVData> features) {
        //Group each item into a list map
        Map<String, List<FVData>> map = new TreeMap<String, List<FVData>>();
        for (FVData feature : features) {
            String key = feature.getType();
            List<FVData> items = map.get(key);
            if (items == null) {
                items = new ArrayList<FVData>();
                map.put(key, items);
            }
            items.add(feature);
        }

        //Convert map into grouped dataset
        BasicGroupedDataset<FVData> dataset = new BasicGroupedDataset<FVData>();
        for (Map.Entry<String, List<FVData>> keyval : map.entrySet()) {
            dataset.put(keyval.getKey(), new ListBackedDataset<FVData>(keyval.getValue()));
        }

        return dataset;
    }

    public static AbstractDenseSIFT<FImage> getSift() {
        DenseSIFT dsift = new DenseSIFT(5, 7);
        return new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
    }
}
