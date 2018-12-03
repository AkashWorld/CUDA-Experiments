TARGET = main
CC = nvcc
CUFLAGS = -lcudart
CFLAGS = -std=c++11
ARCH = -arch=sm_21

all: $(TARGET)

$(TARGET): mm_$(TARGET).cpp
	$(CC) -o $(TARGET) mm_$(TARGET).cu $(CUFLAGS) $(CFLAGS) $(ARCH)

clean:
	rm $(TARGET)

run:
	@$(MAKE) && ./$(TARGET)
