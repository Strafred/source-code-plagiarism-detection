import java.util.Scanner;
public class Kasus6L4 {
    public static void main (String[] args) {
        int[] arrNum = new int[10];
        Scanner inp = new Scanner(System.in);
        for (int i = 0; i < 10; i++) {
            System.out.print("Read a number: ");
            arrNum[i] = inp.nextInt();
        }
        Reverse(arrNum);
    }
    private static void Reverse(int[] arr){
        for (int i = 9; i >= 0; i--) {
            System.out.println(arr[i]);
        }
    }
}
