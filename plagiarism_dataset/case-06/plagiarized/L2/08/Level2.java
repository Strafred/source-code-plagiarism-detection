public class Level2 {
    public static void main(String[] args) {
        java.util.Scanner masukkan = new java.util.Scanner(System.in);
        int[] number = new int[10];
        for (int a = 0; a < 10; a++) {
            System.out.print(
                    "Read a number: ");
            number[a] = masukkan.nextInt();
        }
        for (int a = 9; a >= 0; a--) {
            System.out.println(number[a]);
        }
    }
}
