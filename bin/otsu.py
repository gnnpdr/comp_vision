def otsu(histogramm):
    full_pixel_sum = sum(histogramm)
    
    best_threshold = 0
    best_value = -1 
    
    for threshold in range(256):
        left_pixel_brightness = 0
        left_pixel_count = 0
        for i in range(threshold + 1):
            left_pixel_brightness += histogramm[i] * i 
            left_pixel_count += histogramm[i] 
        
        right_pixel_brightness = 0
        right_pixel_count = 0
        for i in range(threshold + 1, 256): 
            right_pixel_brightness += histogramm[i] * i
            right_pixel_count += histogramm[i]
        
        if left_pixel_count == 0 or right_pixel_count == 0:
            continue
        
        mu_left = left_pixel_brightness / left_pixel_count
        mu_right = right_pixel_brightness / right_pixel_count
        
        w_left = left_pixel_count / full_pixel_sum
        w_right = right_pixel_count / full_pixel_sum
        
        criterion = (mu_left - mu_right) ** 2 * w_left * w_right
        
        if criterion > best_value:
            best_value = criterion
            best_threshold = threshold
    
    return best_threshold