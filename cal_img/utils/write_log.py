from utils import read_ini


class log:
    path = "../result.txt"

    @staticmethod
    def write(result):
        par = read_ini.config.get_all()
        with open(log.path, "a+") as f:
            f.write(str(par))
            f.write("\n")
            f.write(str(result))
            f.write("\n")

    @staticmethod
    def log(first, second):
        with open(log.path, "a+") as f:
            f.write(str(first))
            f.write("\n")
            f.write(second)
            f.write("\n")
