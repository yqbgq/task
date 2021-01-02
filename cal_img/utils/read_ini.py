import configparser


class config:
    path = "../par.ini"
    cf = configparser.ConfigParser()
    cf.read(path)

    @staticmethod
    def get_par(catalogue, key):
        try:
            result = config.cf.get(catalogue, key)
        except configparser.NoOptionError:
            result = False
        return result

    @staticmethod
    def set_par(catalogue, key, value):
        config.cf.set(catalogue, key, str(value))
        config.cf.write(open(config.path, "w+"))

    @staticmethod
    def get_all():
        train_par = config.cf.items("Train-Par")
        pic_par = config.cf.items("Pic-Par")
        return pic_par + train_par
