public class Level4 {
    public static void main(String[] args) {
        java.util.Scanner masukkan = new java.util.Scanner(System.in);
        int number;
        int sisaBagi;
        System.out.print("Enter an integer: ");
        number = masukkan.nextInt();
        while (number != 0) {
            sisaBagi = number % 10;
            System.out.print(sisaBagi);
            number = number / 10;
        }
        System.out.println();
    }
}
