public class Main {
    private static int number;
    public static void main(String[] args) {
        java.util.Scanner input = new java.util.Scanner(System.in);
        System.out.print("Enter an integer: ");
        number = input.nextInt();
        while (number != 0) {
            int remainder = number % 10;
            System.out.print(remainder);
            number = number / 10;
        }
        System.out.println();
    }
}
