# Compiler and flags
CC = clang
CFLAGS = -Wall -Wextra -Ofast -mavx -mfma -g
LDFLAGS = -fsanitize=address,undefined

# Target executable name
TARGET = build/program

# Automatically gather source and object files from all directories
SRCS = $(shell find . -name '*.c')
OBJS = $(patsubst ./%.c, build/%.o, $(SRCS))

# Default rule
all: $(TARGET)

# Link objects to create the final executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile .c files into build/**/*.o
build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule to remove build artifacts
clean:
	rm -rf build/

# Phony targets
.PHONY: all clean
