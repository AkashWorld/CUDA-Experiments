TARGET = main
CC = nvcc
CUFLAGS = -lcudart -lcublas
CFLAGS = -std=c++11 -ccbin g++
ARCH = -arch=sm_35
 all: $(TARGET)
 $(TARGET): mm_$(TARGET).cu
	$(CC) -o $(TARGET) mm_$(TARGET).cu $(CUFLAGS) $(CFLAGS) $(ARCH)
 clean:
	rm $(TARGET)
 run:
	@$(MAKE) && ./$(TARGET)