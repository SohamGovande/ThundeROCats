CXX = hipcc
TARGET = matmul
SOURCE = matmul.hip

.PHONY: $(TARGET)
$(TARGET):
	$(CXX) -O3 -I../../include -fopenmp -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)
