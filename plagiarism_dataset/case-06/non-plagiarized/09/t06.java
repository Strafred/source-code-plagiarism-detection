import java.util.Scanner;
public class t06 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] arrNum = new int[10];
        for (int i = 0; i < 10; i++) {
            System.out.print("Read a number: ");
            arrNum[i] = sc.nextInt();
        }
        for (int i = 9; i > -1; i--) {
            System.out.println(arrNum[i]);
        }
    }
}
