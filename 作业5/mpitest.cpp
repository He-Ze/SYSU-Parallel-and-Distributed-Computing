#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>

#define BUF_SIZE 100000
#define MAX_TASKS 32
#define TEST_TIMES 32

int main(int argc, char* argv[]) {
    char buffer[BUF_SIZE] = {};
    memset(buffer, 'x', BUF_SIZE);
    char host_name[MPI_MAX_PROCESSOR_NAME];
    int host_name_length;
    char host_map[MAX_TASKS][MPI_MAX_PROCESSOR_NAME];
    memset(host_map, 0, MAX_TASKS * MPI_MAX_PROCESSOR_NAME);
    int partner_rank;
    int task_pair[MAX_TASKS] = {};
    memset(buffer, 0, MAX_TASKS);
    double timings[MAX_TASKS][4];
    memset(timings, 0, MAX_TASKS * 3);
    int tag = 1;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size % 2 != 0) 
        exit(1);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Get_processor_name(host_name, &host_name_length);
    MPI_Gather(&host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, &host_map, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (my_rank < size / 2) {
        partner_rank = size / 2 + my_rank;
    } else {
        partner_rank = my_rank - size / 2;
    }
    MPI_Gather(&partner_rank, 1, MPI_INT, &task_pair, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf("There are %d threads totally in the test.\n", size);
        printf("The message size is %d bytes.\n", BUF_SIZE);
        printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    }
    if (my_rank < size / 2) {
        double best = .0, worst = .99E+99, total = .0;
        double total_time = .0;
        for (int i = 0; i < TEST_TIMES; ++i) {
            double nbytes = sizeof(char) * BUF_SIZE;
            double start_time = MPI_Wtime();
            MPI_Send(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD);
            MPI_Recv(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD, &status);
            double end_time = MPI_Wtime();
            double run_time = end_time - start_time;
            double bw = (2 * nbytes) / run_time;
            total += bw;
            best = bw > best ? bw : best;
            worst = bw < worst ? bw : worst;
            total_time += run_time;
        }
        best /= 1000000.0;
        worst /= 1000000.0;
        double avg_bw = (total / 1000000.0) / TEST_TIMES;
        total_time /= TEST_TIMES;
        if (my_rank == 0) {
            timings[0][0] = best;
            timings[0][1] = avg_bw;
            timings[0][2] = worst;
            timings[0][3] = total_time;

            double best_all = .0, worst_all = .0, avg_all = .0;
            double time_all = .0;
            for (int j = 1; j < size / 2; ++j) {
                MPI_Recv(&timings[j], 4, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
            }
            for (int j = 0; j < size / 2; ++j) {
                printf("Test between %d and %d, best bandwidth is %lfMBps, worst bandwidth is %lfMBps, average bandwidth is %lfMBps, time is %lfs.\n", j, task_pair[j], timings[j][0], timings[j][2], timings[j][1], timings[j][3]);
                best_all += timings[j][0];
                avg_all += timings[j][1];
                worst_all += timings[j][2];
                time_all += timings[j][3];
            }
        	printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
            printf("Averagely, best bandwidth is %lfMBps, worst bandwidth is %lfMBps, average bandwidth is %lfMBps, time is %lfs.\n", best_all / (size / 2), worst_all / (size / 2),avg_all / (size / 2),  time_all / (size / 2));
        	printf("------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        } else {
            double tmp_timings[4];
            tmp_timings[0] = best;
            tmp_timings[1] = avg_bw;
            tmp_timings[2] = worst;
            tmp_timings[3] = total_time;
            MPI_Send(tmp_timings, 4, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0; i < TEST_TIMES; ++i) {
            MPI_Recv(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD, &status);
            MPI_Send(&buffer, BUF_SIZE, MPI_CHAR, partner_rank, tag, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}
