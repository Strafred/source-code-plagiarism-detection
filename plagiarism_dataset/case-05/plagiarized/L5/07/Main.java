public class Main {
    public static void main(String[] args) {
        int angka;
        System.out.print("Enter an integer: ");
        java.util.Scanner input = new java.util.Scanner(System.in);
        angka = input.nextInt();
        for (angka = angka; angka != 0; angka = angka / 10) {
            int remainder = angka % 10;
            System.out.print(remainder);
        }
        System.out.println();
    }
}
