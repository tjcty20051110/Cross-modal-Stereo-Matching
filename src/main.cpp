#include "cli/cli.h"

int main(int argc, char* argv[]) {
    cms::CLI cli;
    return cli.run(argc, argv);
}
