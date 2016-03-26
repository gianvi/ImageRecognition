package org.gian.image_recognition.core.extractors;

import org.apache.commons.math.linear.BigMatrix;
import org.apache.commons.math.linear.MatrixUtils;
import org.gian.image_recognition.core.data.FVData;
import org.gian.image_recognition.core.data.KPData;
import org.gian.image_recognition.core.data.LocalFVData;
import org.gian.image_recognition.core.data.LocalImageData;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Fase on 08/03/16.
 */
public class CFIIFExtractor implements FeatureExtractor<DoubleFV, KPData> {

    AbstractDenseSIFT<FImage> sift;
    HardAssigner<byte[], float[], IntFloatPair> assigner;


    private Map<String, Integer> imagesMap;
    private double[][] clusterXImmagineCF;
    private int[] clusterCardinality;

    private double[][] clusterXImmagineNormCF;

    private static int N;
    private double[] clusterIIF;
    private int[] clusterDifferency;

    private double[][] clusterXImmagineCFIIF;
    private double[][] clusterXImmagineNormCFIIF;

    static BigMatrix featuresMatrix;
    static BigMatrix featureNormMatrix;

    public static int NUM_TRAINING_IMAGES_PER_CLASS = 15;
    public static int NUM_TEST_IMAGES_PER_CLASS = 15;
    public static double NUM_PERCENTAGE_CLUSTER = 1;

    public static String SUBPATH_TO_SAVE = "TR"+NUM_TRAINING_IMAGES_PER_CLASS+"_CL"+NUM_PERCENTAGE_CLUSTER+"/";

    public CFIIFExtractor(AbstractDenseSIFT<FImage> sift, HardAssigner<byte[], float[], IntFloatPair> assigner, List<KPData> corpus) {
        this.sift = sift;
        this.assigner = assigner;

        System.out.println("Inizializz. dell'Extractor...");

        //contatore per la fase di training
        N = 0;                               //N tot immagini utile per l'IIF

        this.clusterXImmagineCF = new double[assigner.size()][corpus.size()];  //TODO add the assigner dimension
        this.clusterCardinality = new int[assigner.size()];  //used to normalize CF  (means how many keypoints match this cluster)

        this.clusterXImmagineNormCF = new double[assigner.size()][corpus.size()];  //TODO add the assigner dimension

        this.clusterIIF = new double[assigner.size()];
        this.clusterDifferency = new int[assigner.size()];   //used to calculate IIF (means how many image match this cluster)

        this.clusterXImmagineCFIIF = new double[assigner.size()][corpus.size()];

        this.clusterXImmagineNormCFIIF = new double[assigner.size()][corpus.size()];

        this.imagesMap = new HashMap<String, Integer>();

    }


    //@Override
    public void extractMyFeature(List<KPData> corpus) {//TODO ImageProvider<FImage> provider) {

        //crea mappa PATHIMAGE - INDEX

        //calcola CF per ogni immagine, per ogni cluster
        System.out.println("1) CREA MATRICE CF");
        for (KPData kpdata : corpus) {

            imagesMap.put(kpdata.getPath(), N);

            System.out.println("    START Immagine(" + N + ") = " + kpdata.getPath().substring(kpdata.getPath().length() - 15) + "  => calculateCFIIF");
            //System.out.println("    Extract feature per questa immagine: ("+kpdata.toString()+" DI "+kpdata.getPath().substring(kpdata.getPath().length()-15)+" ) => calculateCF");

            this.calculateCF(kpdata);
            N++;
        }
        System.out.println("1) ClusterXImaagini = CF(c, i)\n");
        for (int c = 0; c < this.clusterXImmagineCF.length; c++) {
            for (int imm = 0; imm < this.clusterXImmagineCF[0].length; imm++) {
                System.out.print(clusterXImmagineCF[c][imm] + "  ");
            }
            System.out.println("\n");
        }


        System.out.println("2) NORMALIZZA CF(c, i) x CC");
        for (int c = 0; c < clusterXImmagineNormCF.length; c++) {//TODO ctrs.length
            for (int ni = 0; ni < clusterXImmagineNormCF[0].length; ni++) {
                try {
                    clusterXImmagineNormCF[c][ni] = new BigDecimal(Double.toString(clusterXImmagineCF[c][ni] / clusterCardinality[c])).setScale(8, RoundingMode.HALF_UP).doubleValue();
                } catch (Exception e) {
                    clusterXImmagineNormCF[c][ni] = 0.0;
                }
            }
        }
        for (int c = 0; c < this.clusterXImmagineNormCF.length; c++) {
            for (int imm = 0; imm < this.clusterXImmagineNormCF[0].length; imm++) {
                System.out.print(clusterXImmagineNormCF[c][imm] + "  ");
            }
            System.out.println("\n");
        }


        System.out.println("3) Conta immaginiXcluster frequency IIF(c)) ");
        for (int c = 0; c < clusterXImmagineCF.length; c++) {//TODO ctrs.length
            //calculate num different images for each cluster and then IIF
            for (int ni = 0; ni < clusterXImmagineCF[0].length; ni++) {
                if (clusterXImmagineCF[c][ni] != 0) {
                    clusterDifferency[c]++;
                }
            }
            try {
                System.out.print("C: " + c + " = log(" + clusterXImmagineCF[0].length + "/" + clusterDifferency[c] + ") ");
                double iif = Math.log((double) clusterXImmagineCF[0].length / clusterDifferency[c]);
                System.out.println(iif);
                clusterIIF[c] = new BigDecimal(Double.toString(iif)).setScale(8, RoundingMode.HALF_UP).doubleValue();
            } catch (Exception e) {
                clusterIIF[c] = 0.0000001;
            }
        }
        for (int c = 0; c < this.clusterIIF.length; c++) {
            System.out.print(clusterIIF[c] + "  ");
        }


        System.out.println("\n\n\n4) MyVECTORFEATURES:   Calcola CF(c,i) x IIF(c)) ");
        for (int c = 0; c < this.clusterXImmagineCFIIF.length; c++) {
            for (int imm = 0; imm < this.clusterXImmagineCFIIF[0].length; imm++) {
                clusterXImmagineCFIIF[c][imm] = new BigDecimal(Double.toString(clusterXImmagineCF[c][imm] * clusterIIF[c])).setScale(8, RoundingMode.HALF_UP).doubleValue();
                System.out.print(clusterXImmagineCFIIF[c][imm] + "  ");
            }
            System.out.println("\n");
        }
        featuresMatrix = MatrixUtils.createBigMatrix(clusterXImmagineCFIIF);


        System.out.println("\n\n\n4) MyNORM_VECTORFEATURES:   Calcola NormCF(c,i) x IIF(c)) ");
        for (int c = 0; c < this.clusterXImmagineNormCFIIF.length; c++) {
            for (int imm = 0; imm < this.clusterXImmagineNormCFIIF[0].length; imm++) {
                clusterXImmagineNormCFIIF[c][imm] = new BigDecimal(Double.toString(clusterXImmagineNormCF[c][imm] * clusterIIF[c])).setScale(8, RoundingMode.HALF_UP).doubleValue();
                System.out.print(clusterXImmagineNormCFIIF[c][imm] + "  ");
            }
            System.out.println("\n");
        }
        featureNormMatrix = MatrixUtils.createBigMatrix(clusterXImmagineNormCFIIF);


        System.out.println("Stampa il modello...");
        this.printModel();

        System.out.println("Salva il modello...");
        //salva le immagini come -- path | type | featVector
        this.storeFeaturesVectorImage(corpus);

        //return CFiifVector;
    }

    private void calculateCF(ImageProvider<FImage> provider) {
        //dammi l'immagine
        FImage immagine = provider.getImage();
        sift.analyseImage(immagine);

        //dammi i keypoints
        LocalFeatureList<ByteDSIFTKeypoint> kps = sift.getByteKeypoints(0.015f);
        int totKpsImm = kps.size();

        for (int c = 0; c < clusterXImmagineCF.length; c++) {//TODO assigner.size()
            for (ByteDSIFTKeypoint kp : kps) {
                if (c == assigner.assign(kp.descriptor) - 1) {
                    clusterXImmagineCF[c][N]++;
                    clusterCardinality[c]++;
                }
            }
            double realCf = ((clusterXImmagineCF[c][N] / totKpsImm) * 100);
            System.out.println("         Il cluster " + c + " contiene " + clusterXImmagineCF[c][N] + "/" + totKpsImm + " => " + "           cf(i, c)=" + realCf + " kps dell'imm " + N);

            //int normCf = realCf/(cardinalita del cluster c)
            clusterXImmagineCF[c][N] = new BigDecimal(Double.toString(realCf)).setScale(8, RoundingMode.HALF_UP).doubleValue();
        }
    }


    private void printModel() {
        ///////////////////////////////////////////////


        File file = new File("tmp/MODEL/CFIIFExtractor.txt");
        file.getParentFile().mkdirs();
        try (
                PrintWriter output = new PrintWriter(file)
        ) {
            String sc = "ClusterCF (occorrenze di I in C)  = M[c][i].....quanti kps dell'imm I sono finiti nel cluster C?" + "\n";
            for (int i = 0; i < clusterXImmagineCF.length; i++) {
                for (int j = 0; j < clusterXImmagineCF[0].length; j++) {
                    sc += clusterXImmagineCF[i][j] + " ";
                }
                sc += "\n";
            }
            sc += "\n\n\n";


            sc += "ClusterNormCF.....quanti kps dell'imm I sono finiti nel cluster C / la cardinalità del Cluster" + "\n";
            for (int i = 0; i < clusterXImmagineNormCF.length; i++) {

                for (int j = 0; j < clusterXImmagineNormCF[0].length; j++) {
                    sc += new BigDecimal(Double.toString(clusterXImmagineNormCF[i][j])) + "  ";
                }
                sc += "\n";
            }
            sc += "\n" + "\n\n";


            sc += "ClusterIIF....lg(N/#imm distinte contenute nel cluster+1)" + "\n";
            for (int i = 0; i < clusterIIF.length; i++) {
                sc += Double.toString(clusterIIF[i]) + " ";
            }
            sc += "\n" + "\n\n";

            sc += "ClusterCardinality: quanti keypoints ci sono in ogni cluster? " + "\n";
            for (int i = 0; i < clusterCardinality.length; i++) {
                sc += clusterCardinality[i] + " ";
            }
            sc += "\n" + "\n\n";

            sc += "ClusterDifferency: quante immagini diverse ci sono in ogni cluster?" + "\n";
            for (int i = 0; i < clusterDifferency.length; i++) {

                sc += clusterDifferency[i] + " ";
            }
            sc += "\n" + "\n\n";


            sc += "Per ogni I, e per ciascun C, il corrispondente peso Wj sara’ il prodotto di:" + "\n" +
                    " NormCF: cluster frequency (percentuale di punti di quella immagine che sono stati mappati nel cluster j) " + "\n" +
                    " IIF: inverse image frequency (log (N / il numero di immagini in cui descrittori mappati in quel cluster sono presenti)" + "\n";
            for (int c = 0; c < clusterXImmagineNormCFIIF.length; c++) {

                for (int j = 0; j < clusterXImmagineNormCFIIF[0].length; j++) {


                    //BigDecimal normCf = new BigDecimal(Double.toString(clusterXImmagineNormCF[c][j])).setScale(2, RoundingMode.HALF_UP);
                    //BigDecimal iif = new BigDecimal(Double.toString(clusterIIF[c])).setScale(2, RoundingMode.HALF_UP);

                    //BigDecimal normCfIif = normCf.multiply(iif);

                    sc += clusterXImmagineNormCFIIF[c][j] + " ";       //normCfIif  + " ";//String.format("%.3f",  )     new java.text.DecimalFormat("#.##").format( clusterXImmagineCF[i][j] ) + "  ";
                }
                sc += "\n";
            }
            sc += "\n" + "\n\n";


            sc += "Per ogni I, e per ciascun C, il corrispondente peso Wj sara’ il prodotto di:" + "\n" +
                    " CF: cluster frequency (percentuale di punti di quella immagine che sono stati mappati nel cluster j) " + "\n" +
                    " IIF: inverse image frequency (log (N / il numero di immagini in cui descrittori mappati in quel cluster sono presenti)" + "\n";
            for (int c = 0; c < clusterXImmagineCFIIF.length; c++) {

                for (int j = 0; j < clusterXImmagineCFIIF[0].length; j++) {


                    //BigDecimal normCf = new BigDecimal(Double.toString(clusterXImmagineNormCFIIF[c][j])).setScale(2, RoundingMode.HALF_UP);
                    //BigDecimal iif = new BigDecimal(Double.toString(clusterIIF[c])).setScale(2, RoundingMode.HALF_UP);

                    //BigDecimal normCfIif = normCf.multiply(iif);

                    sc += clusterXImmagineCFIIF[c][j] + " ";         //normCfIif  + " ";//String.format("%.3f",  )     new java.text.DecimalFormat("#.##").format( clusterXImmagineCF[i][j] ) + "  ";
                }
                sc += "\n";
            }
            sc += "\n" + "\n\n";
            ///////////////////////////////////////////////


            output.println(sc);
            output.close();

        } catch (FileNotFoundException e) {

            e.printStackTrace();
        }
    }


    //TODO flag: 1 = normalizedFeatVector
    //      0 = featuresVector
    private void storeFeaturesVectorImage(List<KPData> trainingSet) {
        //transpose(clusterXImmagineCFIIF);

        List<FVData> features = new ArrayList<FVData>();
        for (KPData kpdata : trainingSet) {
            System.out.println("    END Immagine(" + imagesMap.get(kpdata.getPath()) + ") = " + kpdata.getPath().substring(kpdata.getPath().length() - 15) + "  => storeCFxIIF");
            features.add(new LocalFVData(kpdata.getPath(), kpdata.getType(), new DoubleFV(featuresMatrix.getColumnAsDoubleArray(imagesMap.get(kpdata.getPath())))));
        }
        //Write features to file
        try {
            LocalImageData.writeList("tmp/"+SUBPATH_TO_SAVE+"4myFeatures.txt", features);
        } catch (Exception e) {
            System.out.println("MAMMMMMMM");
        }


        //return null;
        //System.out.println("..ed ora che lo hai estratto salva il featVector:  \n  length: "+fv.length()+" \n  vectorLength: "+fv.getVector().length+"  \n --- Values: 3DECIMAL");
        //features.add(new LocalFVData(kpdata.getPath(), kpdata.getType(), fv));
    }


    private void transpose(double[][] original) {


        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[i].length; j++) {
                System.out.print(original[i][j] + " ");
            }
            System.out.print("\n");
        }
        System.out.print("\n\n matrix transpose:\n");
        // transpose
        double[][] transposed = new double[original[0].length][original.length];
        System.out.println("Original length: " + transposed.length);
        System.out.println("Original length[0]: " + transposed[0].length);
        if (original.length > 0) {
            for (int i = 0; i < original[0].length; i++) {
                for (int j = 0; j < original.length; j++) {
                    System.out.print(original[j][i] + " ");
                }
                System.out.print("\n");
            }
        }
    }


    public DoubleFV getFeature(int imageIndex) {
        DoubleFV fv = new DoubleFV(featuresMatrix.getColumnAsDoubleArray(imageIndex));
        return fv;
    }


    @Override
    public DoubleFV extractFeature(KPData imm) {
        try {
            String path = imm.getPath();
            int index = imagesMap.get(path);
            System.out.println("PATH: " + path + "  INDEX: " + index);
            return new DoubleFV(featuresMatrix.getColumnAsDoubleArray(index));
        } catch (Exception e) {
            System.out.println("ATTENZIONE: l'immagine ricercata non è nel modello!");
            return null;
        }
    }
}
