public class Level3 {
    public static void main(String[] args) {
        java.util.Scanner input = new java.util.Scanner(System.in);
        int angka;
        System.out.print("Enter an integer: ");
        angka = input.nextInt();
        reverse(angka);
    }
    public static void reverse(int number) {
        int sisa;
        while (number != 0) {
            sisa = number % 10;
            System.out.print(sisa);
            number = number / 10;
        }
        System.out.println();
    }
}
