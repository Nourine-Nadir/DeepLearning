import cv2
import numpy as np

# Read the image
img = cv2.imread('fish.jpg')

# Convert the image to grayscale (if necessary)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform 2D FFT
fft = np.fft.fft2(gray_img)

# Shift the DC component to the center
fft_shift = np.fft.fftshift(fft)

# Calculate the magnitude spectrum
magnitude_spectrum = np.abs(fft_shift)
log_magnitude = 20 * np.log(magnitude_spectrum + 1)

# Modify the magnitude spectrum
new_magnitude = magnitude_spectrum.copy()
new_magnitude[log_magnitude < 20] = 0  # Adjust threshold as needed

# Display the modified magnitude spectrum
cv2.imshow('Modified Magnitude Spectrum', np.uint8(255 * new_magnitude / np.max(new_magnitude)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a phase spectrum
phase_spectrum = np.angle(fft_shift)

# Combine modified magnitude and original phase
fft_shift_recovered = new_magnitude * np.exp(1j * phase_spectrum)

# Shift the DC component back
fft_recovered = np.fft.ifftshift(fft_shift_recovered)

# Perform inverse FFT
img_recovered = np.real(np.fft.ifft2(fft_recovered))

# Normalize the recovered image for display
img_recovered = np.uint8(255 * (img_recovered - np.min(img_recovered)) / (np.max(img_recovered) - np.min(img_recovered)))
cv2.imwrite('recovered_image.jpg', img_recovered)

cv2.imshow('Image recovered', img_recovered)
cv2.waitKey(0)
cv2.destroyAllWindows()