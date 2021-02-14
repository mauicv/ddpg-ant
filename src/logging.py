import os
import csv


class Logging:
    def __init__(self, headers, params=None, save_loc='default'):
        self.headers = headers
        self.save_loc = save_loc
        if save_loc not in os.listdir():
            with open(f'save/{self.save_loc}/params', 'w') as param_file:
                for param, val in params.items():
                    param_file.write(f"{param}: {val}\n")

            with open(f'save/{self.save_loc}/logs', 'w') as log_file:
                writer = csv.writer(log_file)
                writer.writerow(headers)
        # else:
        #     with open(f'save/{self.save_loc}/logs', 'rw') as param_file:
        #         for param, val in params.items():
        #             param_file.writeline(f"{param}: {val}")

    def log(self, row):
        print('')
        for item, header in zip(row, self.headers):
            print(header, ': ', item)

        with open(f'save/{self.save_loc}/logs', 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(row)
