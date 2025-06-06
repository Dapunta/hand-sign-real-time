def get_eye_ratio(landmarks, eye_indices):
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    top = ((landmarks[eye_indices[1]][1] + landmarks[eye_indices[2]][1]) / 2)
    bottom = ((landmarks[eye_indices[5]][1] + landmarks[eye_indices[4]][1]) / 2)
    eye_width = ((right[0] - left[0])**2 + (right[1] - left[1])**2)**0.5
    eye_height = (bottom - top)
    return eye_height / eye_width