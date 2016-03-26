package org.gian.image_recognition.core.data;


import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;
import org.openimaj.io.ReadWriteable;


public interface ImageData extends ReadWriteable, ImageProvider<FImage> {
    FImage getImage();

    String getType();

    String getPath();
}
