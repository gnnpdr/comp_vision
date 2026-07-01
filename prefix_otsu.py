def otsu(histogramm):
    prefix_list_brightness = [0] * 256
    prefix_list_pixel = [0] * 256

    for i in range(256):
        if i == 0:
            prefix_list_brightness[i] = histogramm[i] * i
            prefix_list_pixel[i] = histogramm[i]
        else:
            prefix_list_brightness[i] = prefix_list_brightness[i - 1] + histogramm[i] * i
            prefix_list_pixel[i] = prefix_list_pixel[i - 1] + histogramm[i]

    best_value = -1
    best_threshold = 0
    
    for threshold in range(256):
        if prefix_list_pixel[threshold] == 0 or (prefix_list_pixel[255] - prefix_list_pixel[threshold]) == 0:
            continue

        w_left = prefix_list_pixel[threshold] / prefix_list_pixel[255]
        mu_left = prefix_list_brightness[threshold] / prefix_list_pixel[threshold]

        w_right = (prefix_list_pixel[255] - prefix_list_pixel[threshold]) / prefix_list_pixel[255]
        mu_right = (prefix_list_brightness[255] - prefix_list_brightness[threshold]) / (prefix_list_pixel[255] - prefix_list_pixel[threshold])

        value = (mu_left - mu_right) ** 2 * w_right * w_left

        if value > best_value:
            best_value = value
            best_threshold = threshold
        
    
    return best_threshold