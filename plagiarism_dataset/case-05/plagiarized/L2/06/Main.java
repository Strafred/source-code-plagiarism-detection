public class Main {
    public static void main(String[] args) {
        System.out.print("Enter an integer: ");
        java.util.Scanner input = new java.util.Scanner(System.in);
        int angka = input.nextInt();
        reverse(angka);
    }
    public static void reverse(int angka) {
        while (angka != 0) {
            int remainder = angka % 10;
            System.out.print(remainder);
            angka = angka / 10;
        }
        System.out.println();
    }
}
