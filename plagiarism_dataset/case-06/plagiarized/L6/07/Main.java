public class Main {
    public static void main(String[] args) {
        int[] angka;
        java.util.Scanner input = new java.util.Scanner(System.in);
        angka = new int[10];
        for (int i = 0; i < 10; i++) {
            System.out.print(
                    "Read a number: ");
            angka[i] = input.nextInt();
        }
        showArray(angka);
    }
    public static void showArray(int[] angka){
        int i = 0;
        while ( i <= 9) {
            System.out.println(angka[i]);
             i++;
        }
    }
}
