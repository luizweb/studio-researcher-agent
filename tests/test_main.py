from main import print_hello

def test_print_hello(capfd):
    # Chama a função
    print_hello()
    # Captura a saída do terminal
    captured = capfd.readouterr()
    # Verifica se a saída é a esperada
    assert captured.out == "Hello from studio-project-template!\n"