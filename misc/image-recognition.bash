#!/bin/bash
#####################################################################################
###                           Image Recognition Script                            ###
#                                                                                   #
# After performing this script, you will obtain the 2 required files for the image  #
# recognition user interface (tmp/centroids.txt nd tmp/features.txt). See README.md #
# to perform the required prerequisites for this script.                            #
#####################################################################################


##########################################
# DEFINE ENVIRONMENT CONSTANTS
##########################################

# Replace these values with your own values!

# Hadoop File System paths - make sure all of these files/folders exist!
HDFS_TRAIN_DIR=/user/hue/image-recognition/imagecl/train
HDFS_TRAIN_MAP=/user/hue/image-recognition/train-map.txt
HDFS_RESULTS_DIR=/user/hue/image-recognition/results

# Local file system paths - make sure all of these files/folders exist!
LFS_JAR_FILE=/home/ict/image-recognition/image-recognition-0.0.1-jar-with-dependencies.jar
LFS_RESULTS_DIR=/home/ict/image-recognition/results

##########################################
# Check pre-requisites
##########################################

echo "checking pre-requisites...";

# Ensure LFS_RESULTS_DIR exists
if ! [ -e ${LFS_JAR_FILE} ]; then
    echo "ERROR: $LFS_JAR_FILE does not exist on local file system";
    exit;
fi

# Ensure LFS_RESULTS_DIR exists
if ! [ -e ${LFS_RESULTS_DIR} ]; then
    echo "ERROR: ${LFS_RESULTS_DIR}/ does not exist on local file system";
    exit;
fi

if [ -e ${LFS_RESULTS_DIR}/keypoints.txt ]; then
    echo "ERROR: ${LFS_RESULTS_DIR}/keypoints.txt already exists on local file system";
    exit;
fi

if [ -e ${LFS_RESULTS_DIR}/centroids.txt ]; then
    echo "ERROR: ${LFS_RESULTS_DIR}/centroids.txt already exists on local file system";
    exit;
fi

if [ -e ${LFS_RESULTS_DIR}/features.txt ]; then
    echo "ERROR: ${LFS_RESULTS_DIR}/features.txt already exists on local file system";
    exit;
fi

# Ensure HDFS_RESULTS_DIR exists
if ! hadoop fs -test -e ${HDFS_RESULTS_DIR}; then
    echo "ERROR: ${HDFS_RESULTS_DIR}/ does not exist on HDFS";
    exit;
fi

# Ensure kpextractor folder does not exist
if hadoop fs -test -e ${HDFS_RESULTS_DIR}/kpextractor/; then
    echo "ERROR: ${HDFS_RESULTS_DIR}/kpextractor/ already exists on HDFS";
    exit;
fi

# Ensure centroids.txt folder does not exist
if hadoop fs -test -e ${HDFS_RESULTS_DIR}/centroids.txt; then
    echo "ERROR: ${HDFS_RESULTS_DIR}/centroids.txt already exists on HDFS";
    exit;
fi

# Ensure if fvextractor folder does not exist
if hadoop fs -test -e ${HDFS_RESULTS_DIR}/fvextractor/; then
    echo "ERROR: ${HDFS_RESULTS_DIR}/fvextractor/ already exists on HDFS";
    exit;
fi


##########################################
# Keypoint extraction stage
##########################################

# Perform keypoint extraction mapreduce task
hadoop jar ${LFS_JAR_FILE} kpextractor -dir ${HDFS_TRAIN_DIR}/ -map ${HDFS_TRAIN_MAP} -o ${HDFS_RESULTS_DIR}/kpextractor/


##########################################
# Training stage
##########################################

# Copy keypoints file to local file system
hadoop fs -copyToLocal ${HDFS_RESULTS_DIR}/kpextractor/part-r-00000 ${LFS_RESULTS_DIR}/keypoints.txt

# Perform training task
java -Xmx2g -jar ${LFS_JAR_FILE} trainer -kp ${LFS_RESULTS_DIR}/keypoints.txt -o ${LFS_RESULTS_DIR}/centroids.txt

##########################################
# Feature vector extraction stage
##########################################

# Copy centroids file to hadoop file system
hadoop fs -copyFromLocal ${LFS_RESULTS_DIR}/centroids.txt ${HDFS_RESULTS_DIR}/centroids.txt

# Perform feature vector extraction mapreduce task
hadoop jar ${LFS_JAR_FILE} fvextractor -dir ${HDFS_TRAIN_DIR}/ -map ${HDFS_TRAIN_MAP} -cen ${HDFS_RESULTS_DIR}/centroids.txt -o ${HDFS_RESULTS_DIR}/fvextractor/

##########################################
# Wrap-up
##########################################

# Copy feature vector file to local file system
hadoop fs -copyToLocal ${HDFS_RESULTS_DIR}/fvextractor/part-r-00000 ${LFS_RESULTS_DIR}/features.txt

# Indicate end of script
echo "Image recognition process complete! You can now download the following files to recognize images:"
echo ${LFS_RESULTS_DIR}/centroids.txt
echo ${LFS_RESULTS_DIR}/features.txt