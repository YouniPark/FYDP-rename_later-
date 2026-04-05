FRAME_RATE = 30

def convert_to_seconds(time_string):
    """
    Convert time string in format mm:ss:ff to total seconds with decimals.
    
    Args:
        time_string: String in format "mm:ss:ff" (minutes:seconds:frames)
    
    Returns:
        Float representing total seconds
    """
    try:
        parts = time_string.split(':')
        if len(parts) != 3:
            raise ValueError("Format must be mm:ss:ff")
        
        minutes = int(parts[0])
        seconds = int(parts[1])
        frames = int(parts[2])
        
        if frames >= FRAME_RATE:
            raise ValueError(f"Frames must be between 0 and {FRAME_RATE - 1}")
        
        total_seconds = minutes * 60 + seconds + frames / FRAME_RATE
        return total_seconds
    except ValueError as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"Video Time Converter (Frame Rate: {FRAME_RATE} fps)")
    print("Enter time in format mm:ss:ff (or 'quit' to exit)")
    
    while True:
        user_input = input("\nEnter time: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        
        result = convert_to_seconds(user_input)
        if result is not None:
            print(f"{result:.4f}")

if __name__ == "__main__":
    main()
