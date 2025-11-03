class Motor:
    def __init__(self, potencia):
        self.potencia = potencia  # atributo

    def ligar(self):  # método
        print(f"Motor ligado com {self.potencia} W de potência.")


class BombaHidraulica(Motor):
    def bombear(self, vazao):
        print(f"Bombando fluido a {vazao} L/s usando {self.potencia} W.")


# Uso dos objetos
bomba = BombaHidraulica(potencia=1500)
bomba.ligar()
bomba.bombear(vazao=12)
