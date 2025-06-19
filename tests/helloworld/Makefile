CXX = hipcc
TARGET = helloworld
SOURCE = helloworld.hip

$(TARGET): $(SOURCE)
	$(CXX) -O3 -I../../include -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)
